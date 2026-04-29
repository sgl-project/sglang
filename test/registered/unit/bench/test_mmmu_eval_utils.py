import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _load_mmmu_eval_utils():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "benchmark" / "mmmu" / "eval_utils.py"
    module_name = "_test_mmmu_eval_utils"

    stub_modules = {
        "data_utils": _build_data_utils_stub(),
        "datasets": _build_datasets_stub(),
        "numpy": _build_numpy_stub(),
        "torch": types.ModuleType("torch"),
        "tqdm": _build_tqdm_stub(),
    }
    previous_modules = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module

    return module


def _build_data_utils_stub():
    module = types.ModuleType("data_utils")
    module.CAT_SHORT2LONG = {}
    module.DOMAIN_CAT2SUB_CAT = {}

    def _unused(*args, **kwargs):
        raise AssertionError("Unexpected data_utils call in MMMU parser unit test")

    module.construct_prompt = _unused
    module.load_yaml = _unused
    module.process_single_sample = _unused
    module.save_json = _unused
    return module


def _build_datasets_stub():
    module = types.ModuleType("datasets")

    def _unused(*args, **kwargs):
        raise AssertionError("Unexpected datasets call in MMMU parser unit test")

    module.concatenate_datasets = _unused
    module.load_dataset = _unused
    return module


def _build_numpy_stub():
    module = types.ModuleType("numpy")
    module.argmax = lambda values: max(range(len(values)), key=values.__getitem__)
    return module


def _build_tqdm_stub():
    module = types.ModuleType("tqdm")
    module.tqdm = lambda iterable=None, *args, **kwargs: iterable
    return module


class TestMMMUEvalUtils(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_utils = _load_mmmu_eval_utils()

    def test_default_response_answer_regex_captures_multiline_response(self):
        response = "Based on the diagram, compare the labeled points.\nAnswer: B"

        answer = re.search(self.eval_utils.EvalArgs.response_answer_regex, response)

        self.assertIsNotNone(answer)
        self.assertEqual(answer.group(1), response)

    def test_default_regex_extraction_preserves_multiline_answer_for_processing(self):
        response = "Based on the diagram, compare the labeled points.\nAnswer: B"
        sample = self._multiple_choice_sample(response)
        answer = re.search(self.eval_utils.EvalArgs.response_answer_regex, response)
        answer_dict = {}
        out_samples = {}
        previous_random_choice = self.eval_utils.random.choice
        self.eval_utils.random.choice = lambda choices: "A"
        try:
            self.eval_utils.process_result(
                answer.group(1).strip() if answer else response,
                sample,
                answer_dict,
                out_samples,
            )
        finally:
            self.eval_utils.random.choice = previous_random_choice

        self.assertEqual(out_samples["sample-1"]["pred_ans"], "B")

    def test_parse_multi_choice_prefers_explicit_answer_marker_after_copied_options(
        self,
    ):
        response = (
            "The options are:\n"
            "(A) red\n"
            "(B) blue\n"
            "(C) green\n"
            "(D) yellow\n"
            "Answer: B"
        )

        pred_ans = self.eval_utils.parse_multi_choice_response(
            response, ["A", "B", "C", "D"], self._index_to_answer()
        )

        self.assertEqual(pred_ans, "B")

    def test_parse_multi_choice_prefers_final_standalone_letter_after_copied_options(
        self,
    ):
        response = (
            "The options are:\n"
            "(A) red\n"
            "(B) blue\n"
            "(C) green\n"
            "(D) yellow\n"
            "The diagram rules out the other labels.\n"
            "**B**"
        )

        pred_ans = self.eval_utils.parse_multi_choice_response(
            response, ["A", "B", "C", "D"], self._index_to_answer()
        )

        self.assertEqual(pred_ans, "B")

    def test_parse_multi_choice_prefers_latest_explicit_answer(self):
        response = "Initial thought: Answer: A\nAfter checking the image again:\n**B**"

        pred_ans = self.eval_utils.parse_multi_choice_response(
            response, ["A", "B", "C", "D"], self._index_to_answer()
        )

        self.assertEqual(pred_ans, "B")

    def _multiple_choice_sample(self, response):
        return {
            "id": "sample-1",
            "question_type": "multiple-choice",
            "all_choices": ["A", "B", "C", "D"],
            "index2ans": self._index_to_answer(),
            "answer": "B",
            "original_response": response,
        }

    def _index_to_answer(self):
        return {"A": "red", "B": "blue", "C": "green", "D": "yellow"}


if __name__ == "__main__":
    unittest.main()

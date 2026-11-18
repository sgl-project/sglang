import math
import unittest
from types import SimpleNamespace

from transformers import PretrainedConfig

from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestClassifyProbabilities(CustomTestCase):
    def _handler(self, problem_type):
        config = PretrainedConfig(
            problem_type=problem_type,
            id2label={0: "first", 1: "second"},
        )
        manager = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            server_args=None,
            served_model_name="classifier",
            model_path="classifier",
        )
        return OpenAIServingClassify(manager, None)

    def test_multilabel_returns_independent_probabilities(self):
        handler = self._handler("multi_label_classification")
        response = handler._build_classify_response(
            [
                {"embedding": [2.0, 2.0], "meta_info": {"prompt_tokens": 7}},
                {"embedding": [-1000.0, 1000.0]},
            ]
        ).model_dump()
        expected = 1.0 / (1.0 + math.exp(-2.0))
        for probability in response["data"][0]["probs"]:
            self.assertAlmostEqual(probability, expected, places=6)
        self.assertEqual(response["data"][0]["label"], "first")
        self.assertEqual(response["data"][1]["probs"], [0.0, 1.0])
        self.assertEqual(response["data"][1]["label"], "second")
        self.assertEqual(response["usage"]["prompt_tokens"], 7)

    def test_single_label_and_unspecified_models_keep_softmax(self):
        for problem_type in (None, "single_label_classification"):
            with self.subTest(problem_type=problem_type):
                handler = self._handler(problem_type)
                response = handler._build_classify_response(
                    [{"embedding": [2.0, 2.0]}]
                ).model_dump()
                self.assertEqual(response["data"][0]["probs"], [0.5, 0.5])
                self.assertEqual(response["data"][0]["label"], "first")


if __name__ == "__main__":
    unittest.main()

from types import SimpleNamespace

from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k
from sglang.test.test_utils import CustomTestCase


class GSM8KMixin:
    accuracy: float

    def test_gsm8k(self: CustomTestCase):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], self.accuracy)

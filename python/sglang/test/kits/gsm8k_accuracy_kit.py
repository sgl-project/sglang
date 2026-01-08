from types import SimpleNamespace
from typing import Optional

import requests

from sglang.test.few_shot_gsm8k import run_eval as run_eval_gsm8k


class GSM8KMixin:
    gsm8k_accuracy_thres: float
    gsm8k_accept_length_thres: Optional[float] = None
    gsm8k_num_questions: int = 200
    gsm8k_parallel: int = 128

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=self.gsm8k_num_questions,
            max_new_tokens=512,
            parallel=self.gsm8k_parallel,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_accuracy_thres)

        if self.gsm8k_accept_length_thres is not None:
            server_info = requests.get(self.base_url + "/server_info")
            avg_spec_accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(avg_spec_accept_length, self.gsm8k_accept_length_thres)

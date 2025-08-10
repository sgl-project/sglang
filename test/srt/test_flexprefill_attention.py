"""
FROM RULER TASKS
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ruler/README.md
"""

import os
import random
import re
import unittest
from types import SimpleNamespace
from typing import Optional

import pandas

from sglang.srt.utils import kill_process_tree
from sglang.test import simple_eval_common as common
from sglang.test.run_eval import run_eval
from sglang.test.simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

os.environ["OPENAI_API_KEY"] = "EMPTY"

try:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
except (ImportError, ModuleNotFoundError):
    print("\nPlease install lm-eval via `pip install lm-eval[longtxt,api,ruler]`")
    raise


class TestFlexPrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ["SGL_USE_FLEXPREFILL"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "fa3", "--chunked-prefill-size", "-1"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_niah(self):
        metadata = {
            "pretrained": DEFAULT_MODEL_NAME_FOR_TEST,
            "max_seq_lengths": [32768],
        }

        tasks_list = [
            "niah_multikey_1",
        ]
        task_manager = TaskManager(metadata=metadata)

        results = evaluator.simple_evaluate(
            model="openai-chat-completions",
            model_args="pretrained={},base_url={}/v1/chat/completions".format(
                DEFAULT_MODEL_NAME_FOR_TEST, DEFAULT_URL_FOR_TEST
            ),
            tasks=tasks_list,
            num_fewshot=None,
            batch_size=1,
            max_batch_size=1,
            device="cuda",
            limit=10,
            samples=None,
            task_manager=task_manager,
            apply_chat_template=True,
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            fewshot_random_seed=42,
            verbosity="INFO",
        )

        for each in results["results"]:
            print("{} -> {}".format(each, results["results"][each]))

        self.assertGreaterEqual(
            results["results"]["niah_multikey_1"]["32768,none"], 0.7
        )


if __name__ == "__main__":
    unittest.main()

import io
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import download_and_cache_file, read_jsonl

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-large")

TARGET_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_4b_block7"
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
NUM_PROMPTS = 32
NUM_SEQUENTIAL = 4
MAX_NEW_TOKENS = 256

WALK_ON_LOG = "DSpark int8 markov walk on"
FOLDED_LOG = (
    "DSpark draft proposal (greedy + sampling) folded into the draft cuda graph"
)

# Greedy rows ride in the same batch as top-k / top-p rows. No min_p: the
# deterministic-inference sampler asserts against min_p with a sampling seed.
MIXED_SAMPLING_PARAMS = [
    {"temperature": 0.0},
    {"temperature": 1.0},
    {"temperature": 1.0, "top_k": 20},
    {"temperature": 1.0, "top_p": 0.9},
    {"temperature": 0.7, "top_k": 20, "top_p": 0.9},
]


def _prompts() -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    lines = list(read_jsonl(download_and_cache_file(GSM8K_URL)))[:NUM_PROMPTS]
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": line["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for line in lines
    ]


def _launch(*, int8_walk: bool, log: io.StringIO):
    # --cuda-graph-max-bs-decode 32 keeps the whole batch on the folded graph path.
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "fa3",
        "--speculative-draft-attention-backend",
        "fa3",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        DRAFT_MODEL,
        "--cuda-graph-max-bs-decode",
        str(NUM_PROMPTS),
        "--mem-fraction-static",
        "0.7",
        "--page-size",
        "1",
        "--cuda-graph-backend-prefill=disabled",
        "--enable-deterministic-inference",
    ]
    with (
        envs.SGLANG_DSPARK_OPT_INT8_MARKOV_WALK.override(int8_walk),
        envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"),
    ):
        return popen_launch_server(
            TARGET_MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(log, log),
        )


def _generate(prompts: list[str], sampling_params: list[dict]) -> list[dict]:
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={"text": prompts, "sampling_params": sampling_params},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _greedy_outputs(prompts: list[str]) -> list[str]:
    """Batch of NUM_PROMPTS, then NUM_SEQUENTIAL prompts at bs = 1; under
    deterministic inference both are batch-invariant."""
    params = {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS}
    batch = _generate(prompts, [params] * len(prompts))
    single = [_generate([p], [params])[0] for p in prompts[:NUM_SEQUENTIAL]]
    return [out["text"] for out in batch + single]


def _mixed_accept_length(prompts: list[str]) -> float:
    params = [
        MIXED_SAMPLING_PARAMS[i % len(MIXED_SAMPLING_PARAMS)]
        | {"max_new_tokens": MAX_NEW_TOKENS}
        for i in range(len(prompts))
    ]
    outputs = _generate(prompts, params)
    completion = sum(out["meta_info"]["completion_tokens"] for out in outputs)
    verify_ct = sum(out["meta_info"]["spec_verify_ct"] for out in outputs)
    return completion / verify_ct


def _serve(*, int8_walk: bool, prompts: list[str]) -> tuple[list[str], float, str]:
    log = io.StringIO()
    process = _launch(int8_walk=int8_walk, log=log)
    try:
        return _greedy_outputs(prompts), _mixed_accept_length(prompts), log.getvalue()
    finally:
        terminate_and_kill_process_tree(process)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "the int8 markov-walk kernels are sm_90 only",
)
class TestDSparkInt8MarkovWalk(CustomTestCase):
    """SGLANG_DSPARK_OPT_INT8_MARKOV_WALK must serve the folded draft graph
    without changing greedy output or the sampling accept length."""

    @classmethod
    def setUpClass(cls):
        prompts = _prompts()
        cls.default_greedy, cls.default_accept_length, cls.default_log = _serve(
            int8_walk=False, prompts=prompts
        )
        cls.walk_greedy, cls.walk_accept_length, cls.walk_log = _serve(
            int8_walk=True, prompts=prompts
        )

    def test_walker_attached_to_folded_graph(self):
        """Guards the identity test against a silent fallback to the default walk."""
        self.assertIn(WALK_ON_LOG, self.walk_log)
        self.assertNotIn(WALK_ON_LOG, self.default_log)
        self.assertIn(FOLDED_LOG, self.walk_log)
        self.assertIn(FOLDED_LOG, self.default_log)

    def test_greedy_output_matches_default(self):
        """Lossless: the int8 draft may differ at near-ties, the output may not."""
        mismatches = [
            i
            for i, (expected, walk) in enumerate(
                zip(self.default_greedy, self.walk_greedy)
            )
            if expected != walk
        ]
        self.assertEqual(mismatches, [])

    def test_mixed_sampling_accept_length(self):
        """The verifier rebuilds q from corrected_out; a q that is not the sampled
        distribution moves the sampling accept length far either way."""
        print(f"{self.default_accept_length=:.3f} {self.walk_accept_length=:.3f}")
        ratio = self.walk_accept_length / self.default_accept_length
        self.assertLess(abs(ratio - 1.0), 0.1)


if __name__ == "__main__":
    unittest.main()

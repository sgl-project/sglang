"""End-to-end decode-graph coverage for linear and tree UNO decoding.

On XPU it runs on the intel_xpu attention backend with XPU decode graphs.
The test runs both modes on the same prompts. Linear UNO alternates
LoRA-draft and clean-target variants in one graph runner. Tree UNO uses a
private LoRA-draft runner before native EAGLE tree verification. Besides the
generation contract, short greedy comparisons guard lossless output parity
with autoregressive decoding, and stochastic requests guard nontrivial TPF for
both linear and tree sampling on a small, fixed GSM8K sample.
"""

import os
import unittest
from typing import NamedTuple
from unittest.mock import patch

import requests
from huggingface_hub import snapshot_download

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(
    est_time=480,
    stage="extra-a",
    runner_config="1-gpu-large",
)
register_xpu_ci(est_time=900, suite="nightly-xpu-kernel-main-1-gpu", nightly=True)

_is_xpu = is_xpu()

MODEL = "Qwen/Qwen3-8B"
DEFAULT_UNO_LORA_REPO = "s-sahoo/uno-qwen3-8B"
UNO_ADAPTER_FILES = (
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
)
LORA_PATH_ENV = "SGLANG_TEST_UNO_LORA_PATH"
MAX_NEW_TOKENS = 128
# AR decode and UNO verification use different kernel shapes, so compare a
# bounded greedy prefix instead of requiring full-sequence bitwise identity.
PARITY_TOKENS = 32
# XPU bf16 GEMMs are not batch-invariant, so AR's one-row decode and UNO's
# F-row verify can flip a near-tie; AR alone flipped at a 0.25 gap on B60.
# Same bound as SpecServerKit's decode-vs-prefill logprob_match_delta.
XPU_NEAR_TIE_LOGPROB_GAP = 0.5
# One LoRA draft forward plus one clean verification forward.
FORWARDS_PER_UNO_CYCLE = 2
PROMPTS = (
    (
        "Question: Janet's ducks lay 16 eggs per day. She eats three for "
        "breakfast every morning and bakes muffins for her friends every day "
        "with four. She sells the remainder at the farmers' market daily for "
        "$2 per fresh duck egg. How much in dollars does she make every day "
        "at the farmers' market?\nAnswer:"
    ),
    (
        "Question: A robe takes 2 bolts of blue fiber and half that much "
        "white fiber. How many bolts in total does it take?\nAnswer:"
    ),
    (
        "Question: Josh decides to try flipping a house. He buys a house for "
        "$80,000 and then puts in $50,000 in repairs. This increased the value "
        "of the house by 150%. How much profit did he make?\nAnswer:"
    ),
)


class _UnoConfig(NamedTuple):
    name: str
    speculative_num_steps: int
    speculative_eagle_topk: int
    speculative_num_draft_tokens: int


LINEAR_CONFIG = _UnoConfig(
    name="linear",
    speculative_num_steps=1,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=8,  # F = 8
)
TREE_CONFIG = _UnoConfig(
    name="tree",
    speculative_num_steps=7,  # F = 8
    speculative_eagle_topk=16,
    speculative_num_draft_tokens=8,  # Q = 8
)


def _resolve_uno_lora_path() -> str:
    if configured_path := os.environ.get(LORA_PATH_ENV):
        return configured_path

    snapshot_path = snapshot_download(
        repo_id=DEFAULT_UNO_LORA_REPO,
        allow_patterns=list(UNO_ADAPTER_FILES),
    )
    return os.path.join(snapshot_path, "adapter")


class TestUnoLoraPathResolution(unittest.TestCase):
    def test_uses_explicit_override_without_downloading(self):
        with (
            patch.dict(os.environ, {LORA_PATH_ENV: "/tmp/uno-adapter"}),
            patch(f"{__name__}.snapshot_download") as download,
        ):
            self.assertEqual(_resolve_uno_lora_path(), "/tmp/uno-adapter")
        download.assert_not_called()

    def test_resolves_default_adapter_subdirectory(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                f"{__name__}.snapshot_download", return_value="/tmp/uno-snapshot"
            ) as download,
        ):
            self.assertEqual(_resolve_uno_lora_path(), "/tmp/uno-snapshot/adapter")
        download.assert_called_once_with(
            repo_id=DEFAULT_UNO_LORA_REPO,
            allow_patterns=list(UNO_ADAPTER_FILES),
        )


class TestUnoCudaGraph(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.adapter_path = _resolve_uno_lora_path()

    def _platform_args(self) -> list[str]:
        if _is_xpu:
            # The BMG cookbook decode setup. The 24 GB card needs a higher
            # static fraction to hold the 16.4 GB model plus the 1.4 GB UNO
            # LoRA pool.
            return [
                "--attention-backend",
                "intel_xpu",
                "--page-size",
                "64",
                "--cuda-graph-backend-decode",
                "full",
                "--mem-fraction-static",
                "0.88",
            ]
        return [
            "--attention-backend",
            "fa3",
            "--cuda-graph-max-bs-decode",
            str(len(PROMPTS)),
            "--mem-fraction-static",
            "0.7",
            "--page-size",
            "1",
        ]

    def _server_args(self, config: _UnoConfig | None) -> list[str]:
        args = [
            "--dtype",
            "bfloat16",
            *self._platform_args(),
            "--max-running-requests",
            str(len(PROMPTS)),
            "--disable-radix-cache",
            "--random-seed",
            "17",
        ]
        if config is not None:
            args.extend(
                [
                    "--speculative-algorithm",
                    "UNO",
                    "--uno-lora-path",
                    self.adapter_path,
                    "--speculative-num-steps",
                    str(config.speculative_num_steps),
                    "--speculative-eagle-topk",
                    str(config.speculative_eagle_topk),
                    "--speculative-num-draft-tokens",
                    str(config.speculative_num_draft_tokens),
                ]
            )
        return args

    def _run_ar_reference(self) -> tuple[list[list[int]], list | None]:
        process = None
        try:
            process = popen_launch_server(
                MODEL,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self._server_args(None),
            )
            return self._run_greedy(with_top2=_is_xpu)
        finally:
            if process is not None:
                terminate_and_kill_process_tree(process)

    def _run_config(self, config: _UnoConfig) -> tuple[float, list[list[int]]]:
        process = None
        try:
            process = popen_launch_server(
                MODEL,
                self.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self._server_args(config),
            )
            # UNO rejects returned logprobs, so only AR records its top 2.
            greedy_output_ids, _ = self._run_greedy(with_top2=False)
            tpf = self._run_generation_contract(config)
            return tpf, greedy_output_ids
        finally:
            if process is not None:
                terminate_and_kill_process_tree(process)

    def _run_greedy(self, *, with_top2: bool) -> tuple[list[list[int]], list | None]:
        """Greedy output ids per prompt, plus per-position top-2
        (logprob, token) pairs when ``with_top2``."""
        # A list-valued request can be admitted with different prefill batch
        # shapes across server launches. Run each parity prompt at BS1 so the
        # AR and UNO comparisons use the same execution shape.
        output_ids = []
        top2 = [] if with_top2 else None
        for prompt in PROMPTS:
            request = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": PARITY_TOKENS,
                    "ignore_eos": True,
                },
            }
            if with_top2:
                request.update(return_logprob=True, top_logprobs_num=2)
            response = requests.post(
                self.base_url + "/generate",
                json=request,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            )
            self.assertEqual(response.status_code, 200, response.text)

            result = response.json()
            self.assertIn("output_ids", result, result)
            self.assertEqual(
                len(result["output_ids"]),
                PARITY_TOKENS,
                f"Wrong greedy output length for prompt {prompt!r}",
            )
            output_ids.append(result["output_ids"])
            if with_top2:
                top2.append(
                    [
                        [(logprob, token) for logprob, token, *_ in step]
                        for step in result["meta_info"]["output_top_logprobs"]
                    ]
                )
        return output_ids, top2

    def _assert_ar_parity(
        self,
        mode: str,
        actual: list[list[int]],
        expected: list[list[int]],
        expected_top2: list | None,
    ) -> None:
        """Exact greedy parity, except that with ``expected_top2`` the first
        divergence may be AR's runner-up at a near-tie (see
        XPU_NEAR_TIE_LOGPROB_GAP); later tokens then differ legitimately."""
        for index, (prompt, actual_ids, expected_ids) in enumerate(
            zip(PROMPTS, actual, expected)
        ):
            message = (
                f"{mode} UNO diverged from AR within the first "
                f"{PARITY_TOKENS} tokens for prompt {prompt!r}"
            )
            divergence = next(
                (
                    position
                    for position, (got, want) in enumerate(
                        zip(actual_ids, expected_ids)
                    )
                    if got != want
                ),
                None,
            )
            if divergence is None or expected_top2 is None:
                self.assertEqual(actual_ids, expected_ids, message)
                continue
            (best_logprob, _), (runner_up_logprob, runner_up) = expected_top2[index][
                divergence
            ]
            self.assertEqual(
                actual_ids[divergence],
                runner_up,
                f"{message}: position {divergence} is not AR's runner-up",
            )
            self.assertLessEqual(
                best_logprob - runner_up_logprob,
                XPU_NEAR_TIE_LOGPROB_GAP,
                f"{message}: AR was not near a tie at position {divergence}",
            )

    def _run_generation_contract(self, config: _UnoConfig) -> float:
        server_info = requests.get(self.base_url + "/server_info", timeout=30).json()
        self.assertEqual(
            server_info["speculative_eagle_topk"], config.speculative_eagle_topk
        )
        self.assertEqual(
            server_info["speculative_num_steps"], config.speculative_num_steps
        )
        self.assertEqual(
            server_info["speculative_num_draft_tokens"],
            config.speculative_num_draft_tokens,
        )

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": PROMPTS,
                "sampling_params": {
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "ignore_eos": True,
                },
            },
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        self.assertEqual(response.status_code, 200, response.text)

        results = response.json()
        self.assertEqual(len(results), len(PROMPTS))
        total_completion_tokens = 0
        total_verify_ct = 0
        for result in results:
            self.assertTrue(result["text"].strip())
            meta_info = result["meta_info"]
            self.assertEqual(meta_info["completion_tokens"], MAX_NEW_TOKENS)
            total_completion_tokens += meta_info["completion_tokens"]
            total_verify_ct += meta_info.get("spec_verify_ct", 0)

        self.assertGreater(
            total_verify_ct, 0, f"{config.name} performed no verify steps"
        )
        total_forwards = FORWARDS_PER_UNO_CYCLE * total_verify_ct
        tpf = total_completion_tokens / total_forwards
        self.assertGreater(
            tpf,
            1.5,
            f"{config.name} did not advance beyond autoregressive decoding: {tpf=}",
        )
        return tpf

    def test_ar_parity_and_nontrivial_tpf(self):
        ar_output_ids, ar_top2 = self._run_ar_reference()

        linear_tpf, linear_output_ids = self._run_config(LINEAR_CONFIG)
        self._assert_ar_parity("Linear", linear_output_ids, ar_output_ids, ar_top2)

        tree_tpf, tree_output_ids = self._run_config(TREE_CONFIG)
        self._assert_ar_parity("Tree", tree_output_ids, ar_output_ids, ar_top2)

        print(f"UNO GSM8K sample: {linear_tpf=:.3f}, {tree_tpf=:.3f}")


if __name__ == "__main__":
    unittest.main()

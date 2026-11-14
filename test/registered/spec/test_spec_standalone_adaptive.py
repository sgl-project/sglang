"""STANDALONE adaptive state switching with and without CUDA graphs."""

import json
import re
import tempfile
import unittest
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Two server launches; FA3 matches the existing STANDALONE GPU suite.
register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")


class StandaloneAdaptiveMixin:
    """Use singleton BS slots so switching does not depend on model acceptance."""

    # Ungated, small checkpoints with identical token-to-id mappings. Different
    # hidden sizes also exercise STANDALONE's lack of target hidden-state fusion.
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    draft_model = "Qwen/Qwen2.5-0.5B-Instruct"
    disable_cuda_graph = False
    base_url = DEFAULT_URL_FOR_TEST
    prompt = "Count from 1 to 400, separated by commas. Output only the numbers."

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        temp_dir = tempfile.TemporaryDirectory(prefix="standalone-adaptive-")
        cls.addClassCleanup(temp_dir.cleanup)
        config_path = Path(temp_dir.name) / "adaptive.json"
        config_path.write_text(
            json.dumps(
                {
                    "1": {"candidate_steps": [1]},
                    "2": {"candidate_steps": [3]},
                    "4": {"candidate_steps": [5]},
                    "8": {"candidate_steps": [0]},
                }
            )
        )
        cls.log_path = Path(temp_dir.name) / "server.log"
        server_log = cls.log_path.open("w")
        cls.addClassCleanup(server_log.close)
        args = [
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--attention-backend",
            "fa3",
            "--speculative-algorithm",
            "STANDALONE",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-adaptive",
            "--speculative-adaptive-config",
            str(config_path),
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--cuda-graph-bs-decode",
            "1",
            "2",
            "4",
            "8",
            "--max-running-requests",
            "8",
            "--skip-server-warmup",
            "--decode-log-interval",
            "1",
            "--mem-fraction-static",
            "0.7",
        ]
        if cls.disable_cuda_graph:
            args.append("--disable-cuda-graph")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            # Zero-step must keep draft KV warm; do not inherit the experimental
            # skip-extend switch from a developer's environment.
            env={"SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND": "0"},
            return_stdout_stderr=(server_log, server_log),
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.process)

    def _generate(self, lengths: list[int]) -> list[dict]:
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": [self.prompt] * len(lengths),
                "sampling_params": [
                    {
                        "temperature": 0,
                        "max_new_tokens": length,
                        "ignore_eos": True,
                    }
                    for length in lengths
                ],
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()
        self.assertEqual(len(results), len(lengths))
        for result, length in zip(results, lengths):
            self.assertEqual(result["meta_info"]["completion_tokens"], length)
            self.assertEqual(len(result["output_ids"]), length)
        return results

    def _assert_state(self, steps: int) -> None:
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        states = response.json()["internal_states"]
        self.assertTrue(states)
        for state in states:
            self.assertEqual(state["speculative_num_steps"], steps, state)
            self.assertEqual(state["speculative_num_draft_tokens"], steps + 1, state)

    def test_batch_size_switches_preserve_tokens_and_draft_kv(self):
        """Exercise 1/3/5/0 states, then resume drafting within a live request."""
        # Reset the slot if CustomTestCase retries this test on the same server.
        self._generate([128, 128])
        self._assert_state(3)
        outputs = []
        reference = None
        for batch_size, steps in ((1, 1), (2, 3), (4, 5), (8, 0)):
            with self.subTest(batch_size=batch_size, steps=steps):
                log_start = len(self.log_path.read_text())
                results = self._generate([128] * batch_size)
                # Check the transition itself, not only the final server state:
                # request admission/draining can briefly use smaller BS slots.
                phase_log = self.log_path.read_text()[log_start:]
                self.assertRegex(
                    phase_log,
                    rf"Switch adaptive runtime state: steps \d+ -> {steps},",
                )
                # A graph-enabled launch must actually use the target verify
                # graph for this BS; silently falling back to eager is a failure.
                self.assertRegex(
                    phase_log,
                    rf"Decode batch[^\n]*#running-req: {batch_size},"
                    rf"[^\n]*cuda graph: {not self.disable_cuda_graph},",
                )
                outputs.extend(results)
                if steps == 0:
                    # At BS=8 drafting is disabled: use its greedy output as the
                    # reference for the speculative runs (not an accuracy eval).
                    reference = results[0]["output_ids"]

        self.assertIsNotNone(reference, "No zero-step reference response")

        # Enter the mixed-length batch from a positive state, so the log must
        # prove both entry into zero-step and recovery, not a stale prior state.
        outputs.extend(self._generate([128]))
        self._assert_state(1)
        log_start = len(self.log_path.read_text())
        # Seven requests finish first. The surviving request must carry the KV
        # written at steps=0 into a positive-step draft, without a new prefill.
        recovered = self._generate([128] + [32] * 7)
        switches = re.findall(
            r"Switch adaptive runtime state: steps (\d+) -> (\d+),",
            self.log_path.read_text()[log_start:],
        )
        entered_zero = False
        recovered_from_zero = False
        for old, new in switches:
            if new == "0":
                entered_zero = True
            if entered_zero and old == "0" and new in {"1", "3", "5"}:
                recovered_from_zero = True
        self.assertTrue(
            recovered_from_zero,
            f"No positive-to-zero-to-positive cycle in the live batch: {switches}",
        )
        self._assert_state(1)
        outputs.extend(recovered)
        # Also check a fresh request after recovery.
        outputs.extend(self._generate([128]))
        self._assert_state(1)
        for index, result in enumerate(outputs):
            with self.subTest(output=index):
                tokens = result["output_ids"]
                self.assertEqual(tokens, reference[: len(tokens)])


class TestStandaloneAdaptiveCudaGraph(StandaloneAdaptiveMixin, CustomTestCase):
    pass


class TestStandaloneAdaptiveEager(StandaloneAdaptiveMixin, CustomTestCase):
    disable_cuda_graph = True


if __name__ == "__main__":
    unittest.main()

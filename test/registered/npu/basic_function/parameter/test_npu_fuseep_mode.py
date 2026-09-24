"""Compare selected Ascend FuseEP modes with the unfused W8A8 MoE path.

Default to mode 2 on four Ascend devices. A3 checks teacher-forced logprob
differences. A5 checks GSM8K accuracy for both backends (200 questions,
5-shot, >= 0.90) and records logprob differences as diagnostics. A5 prefers
the last explicit numeric #### answer, falling back to the last number;
the original last-number score is also saved for both backends.
Use SGLANG_TEST_FUSEEP_MODES=1,2
only with an EP size supported by mode 1's per-rank expert limit.
Override SGLANG_TEST_MODEL_PATH to use another
ModelSlim W8A8 MoE checkpoint (for example Qwen3.5-35B-A3B-W8A8). BF16
Qwen3.6-35B-A3B cannot exercise the current SGLang FuseEP weight loader.
Server logs, GSM8K reports, and comparisons are saved under SGLANG_TEST_LOG_DIR.
Set SGLANG_TEST_GSM8K_DATA_PATH to a local GSM8K test.jsonl for offline runs.
"""

import concurrent.futures
import json
import math
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Ascend NPU required")
class TestNpuFuseepMode(CustomTestCase):
    """[Test Category] Parameter; [Test Target] --fuseep-mode=1,2."""

    # Identical teacher-forced prefixes avoid comparing probabilities after
    # free-running generations diverge because of quantization rounding.
    PROMPTS = [
        "Question: What is 2 + 3? Answer: 5.\nQuestion: What is 7 + 8? Answer: 15.",
        "The capital of France is Paris. The capital of Japan is Tokyo.",
        "A shop has 12 apples and sells 5 apples. There are 7 apples left.",
        "Python example:\ndef add(a, b):\n    return a + b\n\nadd(2, 3) returns 5.",
    ]
    ARITHMETIC = [(2, 3, 5), (7, 8, 15), (12, 5, 17), (20, 30, 50)]
    # Use the Qwen3-30B-A3B W8A8 FuseEP model test's sample count and threshold.
    GSM8K_NUM_EXAMPLES = 200
    GSM8K_ACCURACY_THRESHOLD = 0.90

    def setUp(self):
        from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

        self.is_a5 = is_npu_arch35()
        self.model = os.environ.get(
            "SGLANG_TEST_MODEL_PATH", QWEN3_30B_A3B_W8A8_WEIGHTS_PATH
        )
        self.tp_size = int(os.environ.get("SGLANG_TEST_TP_SIZE", "4"))
        self.assertGreaterEqual(self.tp_size, 2, "FuseEP requires expert parallelism")
        self.modes = tuple(
            dict.fromkeys(
                int(mode)
                for mode in os.environ.get("SGLANG_TEST_FUSEEP_MODES", "2").split(",")
            )
        )
        self.assertTrue(self.modes and set(self.modes) <= {1, 2})
        model_dir = Path(self.model)
        config = json.loads((model_dir / "config.json").read_text())
        text_config = config.get("text_config", config)
        self.top_k = text_config["num_experts_per_tok"]
        quant_path = model_dir / "quant_model_description.json"
        self.assertTrue(
            quant_path.is_file(),
            "Use a ModelSlim W8A8 checkpoint; BF16 weights do not have the "
            "expert weight scales required by the current FuseEP integration.",
        )
        quant = json.loads(quant_path.read_text())
        expert_weights = [
            value
            for name, value in quant.items()
            if ".experts." in name
            and name.endswith(".weight")
            and not name.startswith("mtp.")
            and ".mtp." not in name
        ]
        self.assertTrue(expert_weights, "No quantized routed expert weights found")
        self.assertTrue(
            all(value == "W8A8_DYNAMIC" for value in expert_weights),
            "FuseEP requires W8A8_DYNAMIC routed expert weights",
        )
        log_dir = os.environ.get("SGLANG_TEST_LOG_DIR")
        if log_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="sglang-fuseep-")
            self.addCleanup(temporary.cleanup)
            log_dir = temporary.name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

    def _post(self, base_url, endpoint, body):
        response = self.session.post(base_url + endpoint, json=body, timeout=180)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _check_generation(self, base_url):
        result = self._post(
            base_url,
            "/generate",
            {
                "text": self.PROMPTS,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(len(result), len(self.PROMPTS))
        for item in result:
            meta = item["meta_info"]
            self.assertEqual(meta["completion_tokens"], 16)
            self.assertEqual(len(meta["output_token_logprobs"]), 16)
            self.assertEqual(len(meta["input_token_logprobs"]), meta["prompt_tokens"])
            self.assertTrue(item["text"].strip())
            for field in ("input_token_logprobs", "output_token_logprobs"):
                values = [entry[0] for entry in meta[field] if entry[0] is not None]
                self.assertTrue(values)
                self.assertTrue(all(math.isfinite(value) for value in values))
        return result

    def _check_arithmetic(self, base_url):
        def request(problem):
            a, b, expected = problem
            result = self._post(
                base_url,
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"What is {a} + {b}? Answer only with the number.",
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 32,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            answer = result["choices"][0]["message"]["content"]
            self.assertIsInstance(answer, str)
            self.assertEqual(re.findall(r"\d+", answer), [str(expected)], answer)
            return answer

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(request, self.ARITHMETIC))

    def _run_gsm8k(self, base_url, name):
        print(f"[{name}] Running GSM8K: 200 questions, 5-shot", flush=True)
        metrics = run_eval(
            SimpleNamespace(
                base_url=base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                num_examples=self.GSM8K_NUM_EXAMPLES,
                num_shots=5,
                num_threads=128,
                max_tokens=512,
                temperature=0,
                gsm8k_data_path=os.environ.get("SGLANG_TEST_GSM8K_DATA_PATH"),
                gsm8k_answer_mode="last_explicit",
            )
        )
        # run_eval uses the model name for its report, so preserve each backend
        # before the next evaluation overwrites it.
        report = Path(f"/tmp/gsm8k_{self.model.replace('/', '_')}.html")
        shutil.copyfile(report, self.log_dir / f"{name}-gsm8k.html")
        return {
            **{key: float(value) for key, value in metrics.items()},
            "num_examples": self.GSM8K_NUM_EXAMPLES,
            "num_shots": 5,
            "accuracy_threshold": self.GSM8K_ACCURACY_THRESHOLD,
            "answer_mode": "last_explicit",
        }

    def _run_backend(self, mode):
        name = "baseline" if mode is None else f"mode{mode}"
        base_url = f"http://127.0.0.1:{get_open_port()}"
        args = [
            "--trust-remote-code",
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--dtype",
            "bfloat16",
            # The Ascend A3 chunk GDN operator requires BF16 SSM state.
            "--mamba-ssm-dtype",
            "bfloat16",
            "--quantization",
            "modelslim",
            "--tp-size",
            self.tp_size,
            "--mem-fraction-static",
            "0.6",
            "--context-length",
            "4096" if self.is_a5 else "2048",
            "--max-total-tokens",
            "32768" if self.is_a5 else "4096",
            "--max-running-requests",
            "32" if self.is_a5 else "4",
            "--chunked-prefill-size",
            "128",
            "--max-prefill-tokens",
            "128",
            "--cuda-graph-backend-decode",
            "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--disable-radix-cache",
            "--random-seed",
            "42",
            "--moe-a2a-backend",
            "none" if mode is None else "ascend_fuseep",
        ]
        if mode is not None:
            args += ["--fuseep-mode", mode]
        # Mode 1 bounds input tokens per rank; mode 2 bounds RECEIVED expert
        # tokens across ranks, including top-k duplication. Do not reuse 256
        # for mode 2: skewed routing can overflow that receive capacity.
        dispatch_capacity = 256 if mode != 2 else 128 * self.tp_size * self.top_k
        env = {
            **os.environ,
            # Mode 1 double-buffers expert dispatch. Qwen3.5-35B with 256
            # experts, hidden=2048 and maxBs=256 requires at least 513 MiB.
            "HCCL_BUFFSIZE": "1024" if mode == 1 else "200",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": str(dispatch_capacity),
        }
        log_path = self.log_dir / f"{name}.log"
        process = None
        try:
            with log_path.open("w") as log:
                process = popen_launch_server(
                    self.model,
                    base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=args,
                    env=env,
                    return_stdout_stderr=(log, log),
                )
                response = self.session.get(base_url + "/server_info", timeout=30)
                response.raise_for_status()
                server_args = response.json()
                self.assertEqual(
                    server_args["moe_a2a_backend"],
                    "none" if mode is None else "ascend_fuseep",
                )
                if mode is not None:
                    self.assertEqual(server_args["fuseep_mode"], mode)
                    self.assertEqual(server_args["ep_size"], self.tp_size)
                result_path = self.log_dir / f"{name}.json"
                result = {"generation": self._check_generation(base_url)}
                # Preserve generated text/logprobs even if the semantic check
                # fails, so a running server cannot mask incorrect inference.
                result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                result["arithmetic"] = self._check_arithmetic(base_url)
                result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
                if self.is_a5:
                    result["gsm8k"] = self._run_gsm8k(base_url, name)
                    result_path.write_text(
                        json.dumps(result, indent=2, ensure_ascii=False)
                    )
                return result
        except Exception:
            print(log_path.read_text(errors="replace")[-16000:])
            raise
        finally:
            if process is not None:
                kill_process_tree(process.pid)
                process.wait(timeout=30)

    def test_fuseep_modes_match_unfused(self):
        baseline = self._run_backend(None)
        comparisons = {}
        for mode in self.modes:
            with self.subTest(fuseep_mode=mode):
                candidate = self._run_backend(mode)
                errors = []
                for reference, actual in zip(
                    baseline["generation"], candidate["generation"]
                ):
                    ref = reference["meta_info"]["input_token_logprobs"]
                    got = actual["meta_info"]["input_token_logprobs"]
                    self.assertEqual([x[1] for x in ref], [x[1] for x in got])
                    self.assertEqual(
                        [x[0] is None for x in ref], [x[0] is None for x in got]
                    )
                    errors.extend(
                        abs(x[0] - y[0]) for x, y in zip(ref, got) if x[0] is not None
                    )
                self.assertTrue(errors)
                comparisons[f"mode{mode}"] = {
                    "tokens_compared": len(errors),
                    "mean_abs_logprob_diff": float(np.mean(errors)),
                    "max_abs_logprob_diff": max(errors),
                }
                if self.is_a5:
                    comparisons[f"mode{mode}"]["gsm8k"] = {
                        "baseline_score": baseline["gsm8k"]["score"],
                        "candidate_score": candidate["gsm8k"]["score"],
                        "baseline_last_number_score": baseline["gsm8k"][
                            "last_number_score"
                        ],
                        "candidate_last_number_score": candidate["gsm8k"][
                            "last_number_score"
                        ],
                        "answer_mode": "last_explicit",
                        "accuracy_threshold": self.GSM8K_ACCURACY_THRESHOLD,
                    }
                print(json.dumps(comparisons[f"mode{mode}"]))
                # Persist diagnostics before assertions, including failed runs.
                (self.log_dir / "comparison.json").write_text(
                    json.dumps(comparisons, indent=2)
                )
                if self.is_a5:
                    # TP and EP quantize different intermediate partitions and
                    # use different reduction orders. Gate A5 on model accuracy;
                    # retain finite/aligned logprobs and the error diagnostics.
                    for name, result in (
                        ("baseline", baseline),
                        (f"mode{mode}", candidate),
                    ):
                        score = result["gsm8k"]["score"]
                        self.assertTrue(math.isfinite(score), f"{name}: GSM8K {score=}")
                        self.assertGreaterEqual(
                            score,
                            self.GSM8K_ACCURACY_THRESHOLD,
                            f"{name}: GSM8K accuracy {score:.3f} is below "
                            f"{self.GSM8K_ACCURACY_THRESHOLD:.2f}",
                        )
                else:
                    # Preserve A3's numerical regression thresholds.
                    self.assertLess(float(np.mean(errors)), 0.1)
                    self.assertLess(max(errors), 0.6)


if __name__ == "__main__":
    unittest.main()

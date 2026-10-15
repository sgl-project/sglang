"""PLE GSM8K: thinking off, 20 questions per mode; run with python3.

Runs normal PLE inference and paired MTP with PLE disabled/enabled (60 requests).
"""

import json
import os
import resource
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import MODEL_WEIGHTS_DIR
from sglang.test.run_eval import run_eval
from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    GSM8K_URL,
    INVALID,
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl

MODEL = os.path.join(MODEL_WEIGHTS_DIR, "Qwen3.8-Flash-Next")
BACKEND = "pinned"  # Change to "file" to evaluate that backend.
MIN_ACCURACY = 0.95  # Reference CI gate, not a Qwen model-card score.
MAX_TOKENS = 1024
NUM_EXAMPLES = 20  # Scored questions, excluding the five few-shot examples.
MTP_SAMPLES = 20
# Repository layout: <work>/code-mine/sglang/test/manual/ascend/this_file.py.
SCRIPT_DIR = Path(__file__).resolve().parents[5] / "script"


class TestNpuPleOffloadAccuracy(CustomTestCase):
    def mtp_ple_small(self):
        """Compare the same greedy MTP questions with PLE disabled/enabled."""
        data_dir, log_dir = SCRIPT_DIR / "datasets/gsm8k", SCRIPT_DIR / "logs"
        data_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        data_path = download_and_cache_file(GSM8K_URL, str(data_dir / "test.jsonl"))
        rows = list(read_jsonl(data_path))
        self.assertEqual(len(rows), 1319)
        selected = rows[5 : 5 + MTP_SAMPLES]
        prefix = get_few_shot_examples(rows, 5)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": prefix + get_one_example(selected, i, False),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for i in range(len(selected))
        ]
        report_dir = Path(tempfile.mkdtemp(prefix="mtp_ple_small_", dir=log_dir))
        print(
            f"MTP paired test: questions={len(selected)}, reports={report_dir}",
            flush=True,
        )
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
        results = {}
        for mode, switch in (
            ("off", "--no-ple-offload-embedding"),
            ("on", "--ple-offload-embedding"),
        ):
            with (report_dir / f"{mode}.server.log").open("w") as server_log:
                process = popen_launch_server(
                    MODEL,
                    DEFAULT_URL_FOR_TEST,
                    timeout=1800,
                    other_args=(
                        "--trust-remote-code --attention-backend ascend --tp-size 8 "
                        "--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.85 "
                        "--cuda-graph-max-bs-decode 4 --disable-radix-cache --random-seed 42 "
                        "--max-running-requests 4 --max-total-tokens 8192 --max-mamba-cache-size 128 "
                        "--speculative-algorithm EAGLE --speculative-num-steps 1 "
                        "--speculative-eagle-topk 1 --speculative-num-draft-tokens 2"
                    ).split()
                    + [switch, "--ple-offload-backend", "pinned"],
                    return_stdout_stderr=(server_log, server_log),
                )
                results[mode] = []
                try:
                    info_response = requests.get(
                        f"{DEFAULT_URL_FOR_TEST}/server_info", timeout=30
                    )
                    info_response.raise_for_status()
                    info = info_response.json()
                    (report_dir / f"{mode}.config.json").write_text(
                        json.dumps(info, indent=2)
                    )
                    self.assertEqual(info["speculative_algorithm"], "EAGLE")
                    self.assertIs(info["ple_offload_embedding"], mode == "on")
                    for index, row in enumerate(selected):
                        response = requests.post(
                            f"{DEFAULT_URL_FOR_TEST}/generate",
                            json={
                                "text": prompts[index],
                                "sampling_params": {
                                    "temperature": 0,
                                    "max_new_tokens": 1024,
                                    "stop": ["Question", "Assistant:", "<|separator|>"],
                                },
                            },
                            timeout=600,
                        )
                        response.raise_for_status()
                        output = response.json()
                        output.update(
                            index=index + 5,
                            answer=get_answer_value(output["text"]),
                            correct_answer=get_answer_value(row["answer"]),
                        )
                        results[mode].append(output)
                        (report_dir / f"{mode}.json").write_text(
                            json.dumps(results[mode], indent=2)
                        )
                        self.assertGreater(
                            output["meta_info"].get("spec_verify_ct", 0), 0
                        )
                        self.assertEqual(
                            output["meta_info"]["finish_reason"]["type"], "stop"
                        )
                        self.assertNotEqual(output["answer"], INVALID)
                        print(
                            f"MTP {mode}: {index + 1}/{len(selected)}, answer={output['answer']}, correct={output['correct_answer']}",
                            flush=True,
                        )
                finally:
                    kill_process_tree(process.pid)
                    process.wait(timeout=30)
        summary = {
            "num_questions": len(selected),
            "correct": {
                mode: sum(item["answer"] == item["correct_answer"] for item in items)
                for mode, items in results.items()
            },
            "answer_mismatches": [
                left["index"]
                for left, right in zip(results["off"], results["on"])
                if left["answer"] != right["answer"]
            ],
            "token_mismatches": [
                left["index"]
                for left, right in zip(results["off"], results["on"])
                if left["output_ids"] != right["output_ids"]
            ],
        }
        (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        self.assertEqual(summary["answer_mismatches"], [], str(report_dir))
        for mode, correct in summary["correct"].items():
            self.assertGreaterEqual(
                correct / len(selected), MIN_ACCURACY, f"MTP {mode}: {report_dir}"
            )

    def test_mtp_ple_small(self):
        self.mtp_ple_small()

    def test_gsm8k(self):
        self.assertIn(BACKEND, ("pinned", "file"))
        self.assertTrue(os.path.isdir(MODEL), f"Model weights not found: {MODEL}")
        data_dir, log_dir = SCRIPT_DIR / "datasets/gsm8k", SCRIPT_DIR / "logs"
        data_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        data_path = download_and_cache_file(GSM8K_URL, str(data_dir / "test.jsonl"))
        rows = list(read_jsonl(data_path))
        self.assertEqual(len(rows), 1319, "Incomplete GSM8K data; check test.jsonl")
        self.assertTrue(all(row.get("question") and row.get("answer") for row in rows))
        report_dir = Path(tempfile.mkdtemp(prefix=f"gsm8k_{BACKEND}_", dir=log_dir))
        print(
            f"GSM8K: speculative=off, backend={BACKEND}, scored={NUM_EXAMPLES}, reports={report_dir}",
            flush=True,
        )
        if BACKEND == "pinned":
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
        with tempfile.TemporaryDirectory(prefix="ple_gsm8k_", dir=SCRIPT_DIR) as tables:
            process = popen_launch_server(
                MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=1800,
                other_args=(
                    "--trust-remote-code --attention-backend ascend --tp-size 8 "
                    "--mamba-ssm-dtype bfloat16 --mem-fraction-static 0.85 "
                    "--cuda-graph-max-bs-decode 4 --ple-offload-embedding"
                ).split()
                + ["--ple-offload-backend", BACKEND, "--ple-offload-dir", tables],
            )
            try:
                args = SimpleNamespace(
                    model=MODEL,
                    base_url=DEFAULT_URL_FOR_TEST,
                    eval_name="gsm8k",
                    # Raw completion can terminate before answering this
                    # instruction model; use its template with thinking off.
                    api="chat",
                    chat_template_kwargs={"enable_thinking": False},
                    gsm8k_data_path=data_path,
                    num_examples=NUM_EXAMPLES,
                    num_shots=5,
                    num_threads=4,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                )
                metrics = run_eval(args)
                for suffix in ("html", "json"):
                    source = Path(f"/tmp/gsm8k_{MODEL.replace('/', '_')}.{suffix}")
                    shutil.move(str(source), str(report_dir / f"gsm8k.{suffix}"))
                (report_dir / "config.json").write_text(
                    json.dumps(
                        vars(args) | {"backend": BACKEND, "min_accuracy": MIN_ACCURACY},
                        indent=2,
                    )
                )
                self.assertGreaterEqual(metrics["score"], MIN_ACCURACY, str(report_dir))
            finally:
                kill_process_tree(process.pid)
                process.wait(timeout=30)


if __name__ == "__main__":
    unittest.main(verbosity=2)

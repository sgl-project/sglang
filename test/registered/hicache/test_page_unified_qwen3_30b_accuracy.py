"""GSM8K accuracy when Qwen3-30B-A3B L3 pages move between TP2 and PP2.

Three independent servers exercise TP2/PP1 -> TP1/PP2 -> TP2/PP1. Both layouts
use a grid of 24 layers x 2 KV heads, so each local page spans two L3 objects.
The PP server writes a separate namespace from fresh computation for the
reverse direction; re-reading the original TP objects would not test PP writes.

Optional local inputs:
  QWEN3_30B_A3B_MODEL_PATH: local BF16 Qwen3-30B-A3B checkpoint
  SGLANG_TEST_GSM8K_DATA_DIR: directory containing train.jsonl and test.jsonl
  SGLANG_TEST_PAGE_UNIFIED_WORK_DIR: parent directory for logs and L3 objects
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import requests

from sglang.srt.mem_cache.hicache_key_scheme import derive_namespace, namespace_digest
from sglang.srt.mem_cache.utils import get_hash_str, storage_namespace_seed
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import (
    INVALID,
    get_answer_value,
    get_few_shot_examples,
    get_one_example,
)
from sglang.test.kl_test_utils import compare_kl_divergence
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import download_and_cache_file, read_jsonl

register_cuda_ci(est_time=1200, stage="nightly", runner_config="4-gpu-h100")


class TestPageUnifiedQwen330BAccuracy(CustomTestCase):
    model = "Qwen/Qwen3-30B-A3B"
    num_questions = 64
    page_size = 64
    max_new_tokens = 512
    parallel_requests = 4
    # Match the existing BF16 Qwen3-30B-A3B TP/PP completion accuracy gate in
    # test_pp_single_node_extra.py; require the same floor for every cache pass.
    accuracy_floor = 0.74
    max_accuracy_regression = 0.02
    kl_threshold = 0.005
    chunk_bytes = 24 * 2 * 64 * 128 * 2 * 2  # layers, heads, tokens, dim, K/V, BF16

    def _load_questions(self):
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        data_dir = os.environ.get("SGLANG_TEST_GSM8K_DATA_DIR")

        def load_split(split):
            path = (
                Path(data_dir) / f"{split}.jsonl"
                if data_dir
                else download_and_cache_file(
                    "https://raw.githubusercontent.com/openai/grade-school-math/"
                    f"master/grade_school_math/data/{split}.jsonl"
                )
            )
            return list(read_jsonl(str(path)))

        # Training examples supply the demonstrations; all evaluated questions
        # come from the held-out test split.
        prefix = get_few_shot_examples(load_split("train"), 5)
        questions = load_split("test")[: self.num_questions]
        self.assertEqual(len(questions), self.num_questions)
        tokenizer = get_tokenizer(self.model_path)
        cases = []
        for index, question in enumerate(questions):
            prompt = prefix + get_one_example(questions, index, False)
            input_ids = tokenizer.encode(prompt)
            answer = get_answer_value(question["answer"])
            self.assertNotEqual(answer, INVALID)
            self.assertGreater(len(input_ids), 256)
            self.assertLess(len(input_ids) + self.max_new_tokens, 8192)
            cases.append({"input_ids": input_ids, "answer": answer})
        return cases

    @contextmanager
    def _server(self, *, tp_size, pp_size, label):
        base_url = f"http://127.0.0.1:{find_available_port(23437)}"
        args = [
            "--tp-size",
            str(tp_size),
            "--pp-size",
            str(pp_size),
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            "bfloat16",
            "--attention-backend",
            "fa3",
            "--mem-fraction-static",
            "0.8",
            "--context-length",
            "8192",
            "--max-total-tokens",
            "32768",
            "--max-running-requests",
            str(self.parallel_requests),
            "--chunked-prefill-size",
            "2048",
            "--cuda-graph-backend-decode=disabled",
            "--cuda-graph-backend-prefill=disabled",
            "--page-size",
            str(self.page_size),
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-write-policy",
            "write_through",
            "--hicache-io-backend",
            "kernel",
            "--hicache-mem-layout",
            "page_unified",
            "--hicache-storage-key-scheme",
            "unified",
            "--hicache-storage-head-partition",
            "2",
            "--hicache-storage-layer-partition",
            "24",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend-extra-config",
            json.dumps({"max_size": 0, "min_free_space": 0}),
            "--enable-cache-report",
        ]
        log_path = self.work_dir / f"{label}.log"
        print(f"{label}: TP={tp_size}, PP={pp_size}; {log_path}", flush=True)
        with log_path.open("w") as log_file:
            process = popen_launch_server(
                self.model_path,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
                other_args=args,
                env={
                    "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": str(self.storage_dir),
                    "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
                },
                return_stdout_stderr=(log_file, log_file),
            )
            try:
                yield base_url
            finally:
                terminate_and_kill_process_tree(process, wait_timeout=60)

    def _request(self, base_url, input_ids, salt, *, score_start=None):
        scoring = score_start is not None
        response = requests.post(
            f"{base_url}/generate",
            json={
                "input_ids": input_ids,
                "cache_salt": salt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 0 if scoring else self.max_new_tokens,
                    "stop": ["Question", "Assistant:", "<|separator|>"],
                },
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": score_start if scoring else len(input_ids) - 1,
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _salt(namespace, index):
        return f"page-unified-qwen30b-{namespace}-{index}"

    def _run_accuracy(self, base_url, cases, *, namespace, label, expect_storage):
        def generate(index):
            return self._request(
                base_url, cases[index]["input_ids"], self._salt(namespace, index)
            )

        with ThreadPoolExecutor(max_workers=self.parallel_requests) as executor:
            results = list(executor.map(generate, range(len(cases))))
        # Save raw responses before checking them, including cache counters and
        # token logprobs, so a failed accuracy/hit assertion remains diagnosable.
        (self.work_dir / f"{label}.json").write_text(json.dumps(results, indent=2))
        correct = 0
        storage_tokens = 0
        for index, (case, result) in enumerate(zip(cases, results)):
            meta = result["meta_info"]
            if expect_storage:
                details = meta["cached_tokens_details"]
                self.assertEqual(details["device"], 0, (label, index, details))
                self.assertEqual(details["host"], 0, (label, index, details))
                self.assertEqual(details["storage_backend"], "HiCacheFile")
                expected = (
                    (len(case["input_ids"]) - 1) // self.page_size * self.page_size
                )
                self.assertEqual(details["storage"], expected, (label, index, details))
                self.assertEqual(meta["cached_tokens"], expected)
                storage_tokens += details["storage"]
            else:
                self.assertEqual(meta["cached_tokens"], 0, (label, index, meta))
            correct += get_answer_value(result["text"]) == case["answer"]
        accuracy = correct / len(cases)
        self.report[label] = {
            "correct": correct,
            "questions": len(cases),
            "accuracy": accuracy,
            "storage_tokens": storage_tokens,
        }
        self._save_report()
        print(f"{label}: {correct}/{len(cases)} GSM8K, {storage_tokens} L3 tokens")
        self.assertGreaterEqual(accuracy, self.accuracy_floor, label)
        return results, accuracy

    def _canonical_files(self, cases, namespace):
        digest = namespace_digest(
            derive_namespace(
                model_id=self.model_path,
                dtype="bfloat16",
                page_size=self.page_size,
                rank_replicated=False,
                total_kv_heads=4,
                head_group=2,
                layer_partition=24,
            )
        )
        all_files, sample_files = set(), set()
        for index, case in enumerate(cases):
            count = (len(case["input_ids"]) - 1) // self.page_size * self.page_size
            hashes = get_hash_str(
                case["input_ids"][:count],
                storage_namespace_seed(None, self._salt(namespace, index)),
                self.page_size,
            )
            for page_index, page_hash in enumerate(hashes):
                for layer_start in (0, 24):
                    for head_group in (0, 1):
                        name = (
                            f"{page_hash}_{digest}_L{layer_start}-{layer_start + 24}"
                            f"_H{head_group}.bin"
                        )
                        all_files.add(name)
                        if page_index == len(hashes) - 1:
                            sample_files.add(name)
        return all_files, sample_files

    def _wait_for_backup(self, cases, namespace):
        expected_files, sample_files = self._canonical_files(cases, namespace)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            missing = expected_files - {
                path.name for path in self.storage_dir.glob("*.bin")
            }
            if not missing:
                for name in expected_files:
                    self.assertEqual(
                        (self.storage_dir / name).stat().st_size, self.chunk_bytes
                    )
                # Inspect all canonical coordinates and byte lengths; hash the
                # final prompt page of every question to detect replacement.
                return expected_files, {
                    name: hashlib.sha256(
                        (self.storage_dir / name).read_bytes()
                    ).hexdigest()
                    for name in sample_files
                }
            time.sleep(0.2)
        self.fail(
            f"{namespace}: {len(missing)} canonical objects missing: {sorted(missing)[:4]}"
        )

    def _assert_objects_unchanged(self, snapshot):
        expected_files, hashes = snapshot
        for name in expected_files:
            self.assertEqual((self.storage_dir / name).stat().st_size, self.chunk_bytes)
        for name, expected in hashes.items():
            actual = hashlib.sha256((self.storage_dir / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"Reader replaced writer object {name}")

    def _check_teacher_forced(self, base_url, cases, results, label):
        cached_logprobs, fresh_logprobs = [], []
        replay_results = []
        # Score the same generated continuation, so greedy divergence between
        # TP and PP does not invalidate the numerical comparison. A new salt
        # prevents both imported prompt KV and cached answer KV from being used.
        for index, (case, result) in enumerate(zip(cases[:8], results[:8])):
            entries = result["meta_info"]["output_token_logprobs"][:64]
            self.assertGreater(len(entries), 0)
            token_ids = [entry[1] for entry in entries]
            replay = self._request(
                base_url,
                case["input_ids"] + token_ids,
                f"teacher-forced-{label}-{index}",
                score_start=len(case["input_ids"]) - 1,
            )
            replay_results.append(replay)
            self.assertEqual(replay["meta_info"]["cached_tokens"], 0)
            scored = replay["meta_info"]["input_token_logprobs"][-len(entries) :]
            self.assertEqual([entry[1] for entry in scored], token_ids)
            cached_logprobs.append([entry[0] for entry in entries])
            fresh_logprobs.append([entry[0] for entry in scored])
        (self.work_dir / f"{label}-teacher-forced.json").write_text(
            json.dumps(replay_results, indent=2)
        )
        compare_kl_divergence(
            fresh_logprobs,
            cached_logprobs,
            {self.model: {"kl_div": self.kl_threshold}},
            self.model,
            label,
        )

    def _save_report(self):
        (self.work_dir / "accuracy.json").write_text(json.dumps(self.report, indent=2))

    def test_accuracy_tp_to_pp_and_pp_to_tp(self):
        self.model_path = os.environ.get("QWEN3_30B_A3B_MODEL_PATH", self.model)
        self.work_dir = Path(
            tempfile.mkdtemp(
                prefix="page_unified_qwen30b_",
                dir=os.environ.get("SGLANG_TEST_PAGE_UNIFIED_WORK_DIR"),
            )
        )
        self.storage_dir = self.work_dir / "l3"
        self.storage_dir.mkdir()
        self.report = {
            "model": self.model_path,
            "head_partition": 2,
            "layer_partition": 24,
        }
        succeeded = False
        try:
            cases = self._load_questions()
            (self.work_dir / "questions.json").write_text(json.dumps(cases, indent=2))
            # Two independent writer namespaces, their generated answers, and
            # teacher-forced replays must coexist without LRU eviction.
            token_bytes = self.chunk_bytes * 4 // self.page_size
            required = (
                3
                * sum(len(case["input_ids"]) + self.max_new_tokens for case in cases)
                * token_bytes
            )
            self.assertGreater(shutil.disk_usage(self.storage_dir).free, required)

            with self._server(tp_size=2, pp_size=1, label="tp-writer") as base_url:
                _, tp_accuracy = self._run_accuracy(
                    base_url,
                    cases,
                    namespace="tp",
                    label="tp-fresh",
                    expect_storage=False,
                )
                tp_snapshot = self._wait_for_backup(cases, "tp")

            with self._server(
                tp_size=1, pp_size=2, label="pp-reader-writer"
            ) as base_url:
                pp_cached, pp_cached_accuracy = self._run_accuracy(
                    base_url,
                    cases,
                    namespace="tp",
                    label="tp-to-pp",
                    expect_storage=True,
                )
                self._assert_objects_unchanged(tp_snapshot)
                self._check_teacher_forced(base_url, cases, pp_cached, "tp-to-pp")
                _, pp_accuracy = self._run_accuracy(
                    base_url,
                    cases,
                    namespace="pp",
                    label="pp-fresh",
                    expect_storage=False,
                )
                pp_snapshot = self._wait_for_backup(cases, "pp")

            with self._server(tp_size=2, pp_size=1, label="tp-reader") as base_url:
                tp_cached, tp_cached_accuracy = self._run_accuracy(
                    base_url,
                    cases,
                    namespace="pp",
                    label="pp-to-tp",
                    expect_storage=True,
                )
                self._assert_objects_unchanged(pp_snapshot)
                self._check_teacher_forced(base_url, cases, tp_cached, "pp-to-tp")
            # Compare each restored run with cold inference on the same reader
            # topology; TP/PP reduction differences also affect cold accuracy.
            self.assertGreaterEqual(
                pp_cached_accuracy, pp_accuracy - self.max_accuracy_regression
            )
            self.assertGreaterEqual(
                tp_cached_accuracy, tp_accuracy - self.max_accuracy_regression
            )
            succeeded = True
            print(json.dumps(self.report, indent=2))
        finally:
            self._save_report()
            if succeeded:
                shutil.rmtree(self.work_dir)
            else:
                print(
                    f"Retained accuracy results, server logs and L3 objects at {self.work_dir}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

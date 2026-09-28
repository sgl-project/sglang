"""Real Qwen serving acceptance: native, AFD eager, and whole-role graphs.

Run on three owned CUDA devices (four-device CI runner). Override AFD_E2E_MODEL
with an already cached, admitted Qwen3-30B-A3B or Qwen3-235B-A22B checkpoint.
AFD_E2E_LOG_DIR retains server logs and responses; this test allocates no machines.
"""

import json
import os
import re
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import requests
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.afd_fixture import AFDProcessSpec, AFDServerGroup
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=900, stage="weekly", runner_config="4-gpu-b200")


execution_records = []


@contextmanager
def record_configuration(name):
    record = {"name": name, "status": "running", "started_epoch": time.time()}
    execution_records.append(record)
    try:
        yield
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    else:
        record["status"] = "passed"
    finally:
        record["finished_epoch"] = time.time()


class TestQwen3MoEAFDServing(CustomTestCase):
    def test_native_eager_and_role(self):
        execution_records.clear()
        model = os.environ.get("AFD_E2E_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        backend = os.environ.get("AFD_E2E_ATTENTION_BACKEND", "fa4")
        port = int(os.environ.get("AFD_E2E_PORT", "23450"))
        url = f"http://127.0.0.1:{port}"
        tokenizer = AutoTokenizer.from_pretrained(model)
        questions = [("2 + 3", "5"), ("6 * 7", "42"), ("9 - 4", "5"), ("12 / 3", "4")]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"Calculate {question}. Reply with only the integer answer.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for question, _ in questions
        ] * 2
        decode_prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Count from 1 to 20, separated by commas and spaces. Output only the sequence.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        ] * 8
        common = [
            "--attention-backend",
            backend,
            "--moe-runner-backend",
            "triton",
            "--moe-a2a-backend",
            "none",
            "--kv-cache-dtype",
            "bfloat16",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "8192",
            "--max-running-requests",
            "8",
            "--disable-radix-cache",
            "--decode-log-interval",
            "8",
        ]
        with tempfile.TemporaryDirectory(prefix="afd-e2e-") as temporary:
            root = Path(os.environ.get("AFD_E2E_LOG_DIR", temporary))
            root.mkdir(parents=True, exist_ok=True)

            @contextmanager
            def native():
                with (root / "native.log").open("w") as log:
                    process = popen_launch_server(
                        model,
                        url,
                        timeout=1200,
                        other_args=common
                        + [
                            "--tp-size",
                            "1",
                            "--base-gpu-id",
                            "0",
                            "--cuda-graph-max-bs-decode",
                            "8",
                            "--nccl-port",
                            str(port + 1),
                        ],
                        return_stdout_stderr=(log, log),
                    )
                    try:
                        yield
                    finally:
                        terminate_and_kill_process_tree(process, wait_timeout=90)

            def generate(
                lengths, *, texts=prompts, check_answers=True, ignore_eos=True
            ):
                response = requests.post(
                    url + "/generate",
                    json={
                        "text": texts,
                        "sampling_params": [
                            dict(
                                temperature=0,
                                top_k=1,
                                max_new_tokens=n,
                                ignore_eos=ignore_eos,
                            )
                            for n in lengths
                        ],
                        "return_logprob": True,
                    },
                    timeout=300,
                )
                response.raise_for_status()
                rows = response.json()
                self.assertEqual(len(rows), 8)
                for index, row in enumerate(rows):
                    self.assertEqual(
                        row["meta_info"]["completion_tokens"], lengths[index]
                    )
                    if check_answers:
                        self.assertRegex(
                            row["text"], r"^\s*" + questions[index % 4][1] + r"(?:\D|$)"
                        )
                return rows

            with record_configuration("native"), native():
                generate([8] * 8)
                reference = generate(
                    [8] * 8, texts=decode_prompts, check_answers=False, ignore_eos=False
                )
            (root / "native.json").write_text(json.dumps(reference))

            for attention_lanes in (1, 2):
                for graph in (False, True):
                    name = f"{attention_lanes}a1f-{'role' if graph else 'eager'}"
                    # Both AFD variants use the native graph configuration.
                    config = dict(
                        lanes=1,
                        attention_lanes=attention_lanes,
                        stages=2,
                        attention_backend=backend,
                        rendezvous_port=port + 25,
                        close_timeout_seconds=10,
                    )
                    graph_args = (
                        [
                            "--cuda-graph-backend-prefill",
                            "disabled",
                            "--cuda-graph-backend-decode",
                            "full",
                            "--cuda-graph-bs-decode",
                            "2",
                            "4",
                            "8",
                        ]
                        if graph
                        else ["--disable-cuda-graph"]
                    )
                    a_args = (
                        common
                        + graph_args
                        + [
                            "--tp-size",
                            str(attention_lanes),
                            "--base-gpu-id",
                            "0",
                            "--nccl-port",
                            str(port + 1),
                            "--dist-init-addr",
                            f"127.0.0.1:{port + 10}",
                        ]
                    )
                    if attention_lanes > 1:
                        a_args += [
                            "--dp-size",
                            str(attention_lanes),
                            "--enable-dp-attention",
                        ]
                    specs = [
                        AFDProcessSpec.server(
                            role="attention",
                            model=model,
                            afd_config=config,
                            base_url=url,
                            other_args=a_args,
                        ),
                        AFDProcessSpec.server(
                            role="ffn",
                            model=model,
                            afd_config=config,
                            other_args=common
                            + graph_args
                            + [
                                "--tp-size",
                                "1",
                                "--ep-size",
                                "1",
                                "--base-gpu-id",
                                str(attention_lanes),
                                "--nccl-port",
                                str(port + 2),
                            ],
                        ),
                    ]
                    group = AFDServerGroup(
                        specs,
                        base_url=url,
                        log_dir=root / name,
                        startup_timeout=1200,
                        shutdown_timeout=40,
                    )
                    with self.subTest(topology=name), record_configuration(name), group:
                        answers = generate([8] * 8)
                        actual = generate(
                            [8] * 8,
                            texts=decode_prompts,
                            check_answers=False,
                            ignore_eos=False,
                        )
                        responses = dict(
                            answers=answers, decode=actual, native_decode=reference
                        )
                        (root / name / "responses.json").write_text(
                            json.dumps(responses)
                        )
                        # Compare actual unfinished decode, never forced output after EOS.
                        # Compare probabilities only along matching prefixes.
                        for expected, row in zip(reference, actual):
                            native_lp = expected["meta_info"]["output_token_logprobs"]
                            afd_lp = row["meta_info"]["output_token_logprobs"]
                            self.assertEqual(
                                [v[1] for v in native_lp], [v[1] for v in afd_lp]
                            )
                            for position in (0, 1, 3, 7):
                                self.assertAlmostEqual(
                                    native_lp[position][0],
                                    afd_lp[position][0],
                                    delta=0.1,
                                )
                        tail = generate([4, 4, 3, 3, 2, 2, 1, 1], check_answers=False)
                        group.check_healthy()
                        self._cancel_and_check_health(url, prompts[0])
                        generate([8] * 8)
                        group.check_healthy()
                        responses["tail"] = tail
                        (root / name / "responses.json").write_text(
                            json.dumps(responses)
                        )
                        self._assert_execution(group, attention_lanes, graph)
                    self.assertTrue(
                        all(
                            process.poll() is not None
                            for process in group.processes.values()
                        )
                    )
                    self.assertFalse(group.forced_kill_pids)
                    for role, width in (("attention", attention_lanes), ("ffn", 1)):
                        closes = [
                            json.loads(line.split("AFD_FINAL_USAGE ", 1)[1])
                            for line in group.log_paths[role].read_text().splitlines()
                            if "AFD_FINAL_USAGE " in line
                        ]
                        self.assertEqual(len(closes), width)
                        self.assertTrue(all(not v["failures"] for v in closes))

    def _cancel_and_check_health(self, url, prompt):
        rid = "afd-e2e-cancel"
        with requests.post(
            url + "/generate",
            json=dict(
                text=prompt,
                rid=rid,
                stream=True,
                sampling_params=dict(
                    temperature=0, max_new_tokens=512, ignore_eos=True
                ),
            ),
            stream=True,
            timeout=120,
        ) as stream:
            stream.raise_for_status()
            self.assertTrue(next(stream.iter_lines(chunk_size=1)))
            aborted = requests.post(
                url + "/abort_request", json={"rid": rid}, timeout=30
            )
            aborted.raise_for_status()
        healthy = requests.get(url + "/health_generate", timeout=120)
        healthy.raise_for_status()

    def _assert_execution(self, group, attention_lanes, graph):
        for role, width in (("attention", attention_lanes), ("ffn", 1)):
            latest = {}
            for line in group.log_paths[role].read_text().splitlines():
                if "AFD_GRAPH_USAGE_SNAPSHOT " not in line:
                    continue
                prefix, payload = line.split("AFD_GRAPH_USAGE_SNAPSHOT ", 1)
                rank = re.search(r"(?:TP|FFN)(\d+)", prefix)
                # A TP1 logger need not print a rank label.
                latest[int(rank[1]) if rank else 0] = json.loads(payload)
            self.assertEqual(set(latest), set(range(width)))
            for row in latest.values():
                self.assertEqual(row["typed_eager_failures"], 0)
                self.assertEqual(row["terminal_buckets"], 0)
                self.assertEqual(row["overflow_eager"], 0)
                if graph:
                    self.assertGreater(row["installs"], 0)
                    self.assertGreater(row["replays"], 0)
                else:
                    self.assertEqual(row["installs"], 0)
                    self.assertEqual(row["replays"], 0)


if __name__ == "__main__":
    started = time.time()
    result = unittest.main(exit=False).result
    output = os.environ.get("AFD_E2E_RESULT_FILE")
    if output:
        Path(output).write_text(
            json.dumps(
                dict(
                    status="pass"
                    if result.wasSuccessful() and not result.skipped
                    else "fail",
                    correctness_pass=result.wasSuccessful() and not result.skipped,
                    graph_replay_pass=result.wasSuccessful() and not result.skipped,
                    tests_run=result.testsRun,
                    skipped=len(result.skipped),
                    configurations=[
                        record["name"]
                        for record in execution_records
                        if record["status"] == "passed"
                    ],
                    configuration_attempts=execution_records,
                    planned_configurations=[
                        "native",
                        "1a1f-eager",
                        "1a1f-role",
                        "2a1f-eager",
                        "2a1f-role",
                    ],
                    window_epoch=[started, time.time()],
                )
            )
        )
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)

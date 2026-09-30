"""Real overlap capture checks with delayed results, graph padding and abort."""

import hashlib
import json
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.training_capture_utils import exercise_capture_abort


def exercise_overlap_capture(
    test, *, model_path, directory, samples, responses, cuda_graph
):
    root = Path(directory) / f"overlap-{cuda_graph}"
    root.mkdir()
    config = json.loads(test.capture_path.read_text())
    config.update(journal_directory=str(root / "journal"), max_sample_tokens=176)
    capture_path = root / "capture.json"
    capture_path.write_text(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    launch = test_utils._launch_server_process

    def observed_server(command, *args):
        return launch(
            [sys.executable, "-m", "sglang.test.training_capture_overlap_server"]
            + command[2:],
            *args,
        )

    first_publication = len(test.catalog.publications)
    with patch.object(test_utils, "_launch_server_process", observed_server):
        server = test_utils.popen_launch_server(
            model_path,
            url,
            timeout=240,
            other_args=[
                "--training-capture-config",
                str(capture_path),
                "--skip-server-warmup",
                "--attention-backend",
                "triton",
                "--mem-fraction-static",
                "0.25",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--chunked-prefill-size",
                "128",
                "--cuda-graph-backend-decode",
                "full" if cuda_graph else "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "4",
            ],
        )
    results = []
    publications = []
    prompt = samples[0][1]["token_ids"][: samples[0][0].sequence.prompt_length].tolist()

    def wait_admission(count):
        deadline = time.monotonic() + 20
        while True:
            state = requests.get(url + "/server_info", timeout=10).json()[
                "internal_states"
            ][0]["training_capture"]
            if state["states"].get("available", 0) >= count:
                return
            test.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.05)

    def generate(inputs, params):
        wait_admission(len(inputs) if isinstance(inputs[0], list) else 1)
        response = requests.post(
            url + "/generate",
            json={"input_ids": inputs, "sampling_params": params},
            timeout=120,
        )
        test.assertEqual(response.status_code, 200, response.text)
        value = response.json()
        results.extend(value if isinstance(value, list) else [value])
        state = requests.get(url + "/server_info", timeout=10).json()[
            "internal_states"
        ][0]["training_capture"]
        print(
            json.dumps(
                {"overlap_after_request": state, "expected_samples": len(results)}
            ),
            flush=True,
        )
        test.assertEqual(state["counters"].get("admitted", 0), len(results), state)
        return value

    try:
        for index, baseline in enumerate(responses):
            params = {
                "temperature": 0,
                "max_new_tokens": len(baseline["output_ids"]),
                "ignore_eos": True,
            }
            if index == 2:
                params["logit_bias"] = {"100": 100.0}
            actual = generate(prompt, params)
            test.assertEqual(actual["output_ids"], baseline["output_ids"])
        batched = generate(
            [prompt] * 3,
            [
                {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
                {
                    "temperature": 0,
                    "max_new_tokens": 3,
                    "ignore_eos": True,
                    "logit_bias": {"100": 100.0},
                },
                {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
            ],
        )
        for actual, baseline in zip(
            batched, [responses[1], responses[2], responses[0]], strict=True
        ):
            test.assertEqual(actual["output_ids"], baseline["output_ids"])
        generate(
            [100, 200, 300, 400] * 43,
            {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        for name, params in (
            (
                "eos",
                {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "min_new_tokens": 1,
                    "logit_bias": {str(tokenizer.eos_token_id): 100.0},
                },
            ),
            (
                "grammar",
                {"temperature": 0.8, "max_new_tokens": 16, "regex": "[0-9]{12}"},
            ),
        ):
            actual = generate(prompt, params)
            if name == "eos":
                test.assertEqual(len(actual["output_ids"]), 2)
                test.assertEqual(actual["output_ids"][-1], tokenizer.eos_token_id)
            else:
                test.assertRegex(actual["text"], r"^[0-9]{12}$")
        publications = test.catalog.wait_publications(
            first_publication + len(results), timeout=30
        )[first_publication:]
        references = [
            torch.load(path, weights_only=True)
            for path in sorted((root / "capture-reference").glob("*.pt"))
        ]
        captured = [test.read_sample(publication) for publication in publications]
        test.assertCountEqual(
            [
                tensors["token_ids"][manifest.sequence.prompt_length :].tolist()
                for manifest, tensors in captured
            ],
            [result["output_ids"] for result in results],
        )
        for manifest, tensors in captured:
            check_capture_snapshot(
                test, manifest, tensors, references, capture_mode="autoregressive"
            )
        test.assertTrue(any(row["result_lag"] == 1 for row in references))
        first_slots, remapped = {}, set()
        for item in references:
            for index, slot in enumerate(item["kv_slots"].tolist()):
                key = (item["trace_id"], item["kv_start"] + index)
                if first_slots.setdefault(key, slot) != slot:
                    remapped.add(key)
        test.assertTrue(remapped, "prefix dedup did not exercise slot remapping")
        test.assertTrue(any(row["batch_size"] == 3 for row in references))
        if cuda_graph:
            test.assertTrue(
                any(row["batch_size"] == 3 and row["cuda_graph"] for row in references)
            )
        test.assertTrue(
            any(manifest.sequence.total_length == 176 for manifest, _ in captured)
        )
        test.assertTrue(any(bool(tensors["kv_valid"][-1]) for _, tensors in captured))
        state = requests.get(url + "/server_info", timeout=10).json()[
            "internal_states"
        ][0]["training_capture"]
        test.assertTrue(state["enable_overlap"])
        test.assertGreater(state["counters"].get("overlap_forwards", 0), 0)
        test.assertEqual(
            state["counters"].get("cuda_graph_forwards", 0) > 0, cuda_graph
        )

        abort_id = f"abort-overlap-{cuda_graph}"
        wait_admission(1)
        state = exercise_capture_abort(
            test,
            url=url,
            rid=abort_id,
            prompt=prompt[:8],
            max_new_tokens=168,
        )
        test.assertEqual(
            len(test.catalog.publications), first_publication + len(results)
        )
        test.assertNotIn(
            hashlib.sha256(abort_id.encode()).hexdigest(),
            [manifest.provenance.trace_id for manifest, _ in captured],
        )
        print(
            json.dumps(
                {
                    "overlap_capture": state,
                    "samples": len(captured),
                    "cuda_graph_enabled": cuda_graph,
                    "padded_graph": cuda_graph,
                    "aborted": True,
                    "canonical_remaps": len(remapped),
                }
            ),
            flush=True,
        )
    finally:
        kill_process_tree(server.pid)
        server.wait(timeout=20)
    for publication in publications:
        test.read_sample(publication)

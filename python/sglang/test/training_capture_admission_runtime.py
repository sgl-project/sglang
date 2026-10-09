"""Live admission pause/recovery while a captured request waits for Store writing."""

import json
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils


def exercise_latency_capture(test, *, model_path, directory):
    root = Path(directory)
    journal = root / "latency-journal"
    journal.mkdir()
    pause = journal / "latency.pause"
    config = json.loads(Path(test.capture_path).read_text())
    config.update(
        journal_directory=str(journal),
        max_inflight_samples=2,
        sample_ratio=1.0,
        adaptive={
            "interval_seconds": 0.05,
            "cooldown_seconds": 0.2,
            "latency": {
                "ttft_seconds": 0.5,
                "tpot_seconds": 0.5,
                "min_observations": 1,
                "max_observations": 64,
                "window_seconds": 1.5,
            },
        },
    )
    path = root / "latency-capture.json"
    path.write_text(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    launch = test_utils._launch_server_process

    def controlled_server(command, *args):
        return launch(
            [sys.executable, "-m", "sglang.test.training_capture_admission_server"]
            + command[2:],
            *args,
        )

    def state():
        response = requests.get(url + "/server_info", timeout=10)
        response.raise_for_status()
        return response.json()["internal_states"][0]["training_capture"]

    def wait_for(predicate):
        deadline = time.monotonic() + 20
        while True:
            current = state()
            if predicate(current):
                return current
            if time.monotonic() >= deadline:
                raise AssertionError(current)
            time.sleep(0.03)

    prompt = [23, 41, 99, 100]

    def generate():
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
            },
            timeout=30,
        )
        test.assertEqual(response.status_code, 200, response.text)
        return response.json()["output_ids"]

    first_publication = len(test.catalog.publications)
    process = None
    try:
        with patch.object(test_utils, "_launch_server_process", controlled_server):
            process = test_utils.popen_launch_server(
                model_path,
                url,
                timeout=240,
                other_args=[
                    "--skip-server-warmup",
                    "--enable-metrics",
                    "--skip-tokenizer-init",
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
                    "disabled",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--training-capture-config",
                    str(path),
                ],
            )
        wait_for(lambda current: current["states"].get("available") == 2)
        pause.touch()
        output = generate()
        blocked = wait_for(lambda current: current["admission"]["latency"]["blocked"])
        test.assertEqual(blocked["admission"]["effective_ratio"], 0)
        test.assertTrue(blocked["admission"]["reason"].startswith("latency_"))
        test.catalog.wait_publications(first_publication + 1, timeout=30)
        test.assertEqual(generate(), output)
        test.assertEqual(len(test.catalog.publications), first_publication + 1)
        pause.unlink()
        stale = wait_for(
            lambda current: current["admission"]["latency"]["state"] == "stale"
        )
        test.assertEqual(stale["admission"]["effective_ratio"], 0)
        observation_ct = stale["admission"]["latency"]["metrics"]["ttft"][
            "observations"
        ]
        test.assertEqual(generate(), output)
        recovered = wait_for(
            lambda current: current["admission"]["effective_ratio"] == 1
        )
        test.assertGreater(
            recovered["admission"]["latency"]["metrics"]["ttft"]["observations"],
            observation_ct,
        )
        test.assertEqual(recovered["counters"]["admitted"], 1)
        test.assertGreaterEqual(recovered["counters"]["adaptive_sampled_out"], 2)
        test.assertEqual(generate(), output)
        publications = test.catalog.wait_publications(
            first_publication + 2, timeout=30
        )[first_publication:]
        test.assertEqual(len(publications), 2)
        for publication in publications:
            manifest, tensors = test.read_sample(publication)
            test.assertEqual(tensors["token_ids"].tolist(), prompt + output)
            test.assertEqual(manifest.sequence.response_length, 4)
        from prometheus_client.parser import text_string_to_metric_families

        deadline = time.monotonic() + 10
        while True:
            metrics = requests.get(url + "/metrics", timeout=10)
            metrics.raise_for_status()
            values = {
                (
                    sample.name,
                    sample.labels.get("metric"),
                    sample.labels.get("kind"),
                ): sample.value
                for family in text_string_to_metric_families(metrics.text)
                for sample in family.samples
            }
            ttft = values.get(
                ("sglang:training_capture_latency_observations_total", "ttft", None), 0
            )
            if ttft >= 4:
                break
            if time.monotonic() >= deadline:
                raise AssertionError(values)
            time.sleep(0.1)
        test.assertEqual(
            values[
                ("sglang:training_capture_scheduler_latency_seconds", "ttft", "budget")
            ],
            0.5,
        )
        test.assertEqual(
            values[("sglang:training_capture_latency_control_enabled", None, None)], 1
        )
        print(
            json.dumps(
                {"latency_capture": state(), "samples": 2, "observed_requests": ttft}
            ),
            flush=True,
        )
    finally:
        pause.unlink(missing_ok=True)
        if process is not None:
            kill_process_tree(process.pid)
            process.wait(timeout=20)
    for publication in publications:
        test.read_sample(publication)


def exercise_adaptive_capture(test, *, model_path, directory):
    root = Path(directory)
    journal = root / "adaptive-journal"
    journal.mkdir()
    pause = journal / "writer.pause"
    pause.touch()
    config = json.loads(Path(test.capture_path).read_text())
    config.update(
        journal_directory=str(journal),
        max_inflight_samples=2,
        sample_ratio=1.0,
        adaptive={
            "interval_seconds": 0.05,
            "writer_stall_seconds": 0.2,
            "cooldown_seconds": 0.2,
        },
    )
    path = root / "adaptive-capture.json"
    path.write_text(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    original_launch = test_utils._launch_server_process

    def controlled_server(command, *args):
        return original_launch(
            [sys.executable, "-m", "sglang.test.training_capture_admission_server"]
            + command[2:],
            *args,
        )

    process = None
    first_publication = len(test.catalog.publications)

    def state():
        return requests.get(url + "/server_info", timeout=10).json()["internal_states"][
            0
        ]["training_capture"]

    def wait_for(predicate):
        deadline = time.monotonic() + 20
        while True:
            current = state()
            if predicate(current):
                return current
            if time.monotonic() >= deadline:
                raise AssertionError(current)
            time.sleep(0.05)

    def wait_metrics(predicate):
        from prometheus_client.parser import text_string_to_metric_families

        deadline = time.monotonic() + 20
        while True:
            response = requests.get(url + "/metrics", timeout=10)
            response.raise_for_status()
            samples = [
                sample
                for family in text_string_to_metric_families(response.text)
                for sample in family.samples
                if sample.name.startswith("sglang:training_capture_")
            ]

            def value(name, _samples=samples, **labels):
                matching = [
                    sample
                    for sample in _samples
                    if sample.name == "sglang:training_capture_" + name
                    and all(
                        sample.labels.get(key) == val for key, val in labels.items()
                    )
                ]
                test.assertEqual(len(matching), 1, (name, labels, _samples))
                test.assertIn("model_name", matching[0].labels)
                test.assertEqual(matching[0].labels["tp_rank"], "0")
                return matching[0].value

            if samples and predicate(value):
                return value
            if time.monotonic() >= deadline:
                raise AssertionError(samples)
            time.sleep(0.1)

    def generate(prompt):
        response = requests.post(
            url + "/generate",
            json={
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
            },
            timeout=30,
        )
        test.assertEqual(response.status_code, 200, response.text)
        return response.json()["output_ids"]

    def check_publication(index, prompt, output):
        publication = test.catalog.wait_publications(index + 1, timeout=30)[index]
        manifest, tensors = test.read_sample(publication)
        test.assertEqual(tensors["token_ids"].tolist(), prompt + output)
        test.assertEqual(
            tensors["loss_mask"].tolist(), [0] * len(prompt) + [1] * len(output)
        )
        test.assertEqual(manifest.sequence.response_length, len(output))

    try:
        with patch.object(test_utils, "_launch_server_process", controlled_server):
            process = test_utils.popen_launch_server(
                model_path,
                url,
                timeout=240,
                other_args=[
                    "--skip-server-warmup",
                    "--enable-metrics",
                    "--skip-tokenizer-init",
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
                    "disabled",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--training-capture-config",
                    str(path),
                ],
            )
        wait_for(lambda current: current["states"].get("available") == 2)
        prompt = [23, 41, 99, 100]
        output = generate(prompt)
        paused = wait_for(
            lambda current: (
                current["admission"]["effective_ratio"] == 0
                and current["admission"]["reason"] == "writer_stall"
            )
        )
        test.assertTrue((journal / "writer.started").exists())
        test.assertEqual(paused["states"].get("available"), 1)
        paused_metrics = wait_metrics(
            lambda value: (
                value("sample_ratio", kind="effective") == 0
                and value("reservations", state="writing") == 1
            )
        )
        test.assertGreater(paused_metrics("writer_age_seconds"), 0)
        test.assertEqual(paused_metrics("disabled"), 0)
        test.assertEqual(paused_metrics("reservations", state="available"), 1)
        test.assertEqual(generate(prompt), output)
        test.assertEqual(len(test.catalog.publications), first_publication)
        skipped = state()
        test.assertEqual(skipped["counters"]["admitted"], 1)
        test.assertGreaterEqual(skipped["counters"]["adaptive_sampled_out"], 1)
        pause.unlink()
        check_publication(first_publication, prompt, output)
        wait_for(
            lambda current: (
                current["admission"]["effective_ratio"] == 1
                and current["states"].get("available") == 2
            )
        )
        prompt = [23, 41, 99, 101]
        output = generate(prompt)
        check_publication(first_publication + 1, prompt, output)
        final = wait_for(lambda current: current["states"].get("available") == 2)
        test.assertEqual(final["host_pool"]["quarantined"], 0)
        test.assertEqual(final["counters"]["ready"], 2)
        recovered_metrics = wait_metrics(
            lambda value: (
                value("events_total", event="ready") == 2
                and value("reservations", state="available") == 2
                and value("sample_ratio", kind="effective") == 1
            )
        )
        test.assertEqual(
            recovered_metrics("events_total", event="adaptive_sampled_out"), 1
        )
        test.assertEqual(recovered_metrics("reservations", state="writing"), 0)
        test.assertEqual(recovered_metrics("host_slots", state="quarantined"), 0)
        test.assertEqual(recovered_metrics("writer_age_seconds"), 0)
        test.assertGreater(
            recovered_metrics("admission_adjustments_total", action="pauses"), 0
        )
        print(json.dumps({"adaptive_capture": final, "samples": 2}), flush=True)
    finally:
        pause.unlink(missing_ok=True)
        if process is not None:
            kill_process_tree(process.pid)
            process.wait(timeout=20)

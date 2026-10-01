"""Synthetic KV-draft export and live serving checks, never a trained artifact."""

import json
import socket
import sys
import time
from pathlib import Path
from unittest.mock import patch

import msgspec
import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_target_kv_contract import (
    KVCompatibility,
    KVEncoderConfig,
    KVSequenceContract,
    KVTrainingContract,
    KVValidation,
    TargetKVDraftContract,
)
from sglang.srt.training_capture.protocol import digest_bytes
from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.training_capture_utils import exercise_capture_abort


def export_synthetic_kv_draft(model_path, destination, manifest, tensors):
    root = Path(model_path)
    destination.mkdir()
    config = json.loads((root / "config.json").read_text())
    hidden_size, vocab_size = config["hidden_size"], config["vocab_size"]
    contract = TargetKVDraftContract(
        teacher=manifest.teacher,
        kv=manifest.kv,
        encoder=KVEncoderConfig(hidden_size=hidden_size, rms_norm_eps=1e-6),
        sequence=KVSequenceContract(
            prediction_count=3, input_length=3, mask_token_id=vocab_size - 1
        ),
        training=KVTrainingContract(lambda_tv=0.5),
        compatibility=KVCompatibility(
            sglang_revision="checkout-under-test", specforge_revision="synthetic-test"
        ),
        validation=KVValidation(
            golden_fixture_sha256="0" * 64, parity_rtol=0.03, parity_atol=0.03
        ),
    )
    validation = destination / "validation"
    validation.mkdir()
    fixture_path = validation / "inputs.safetensors"
    save_file(
        {name: value.contiguous() for name, value in tensors.items()}, str(fixture_path)
    )
    contract = msgspec.structs.replace(
        contract,
        validation=msgspec.structs.replace(
            contract.validation,
            golden_fixture_sha256=digest_bytes(fixture_path.read_bytes()),
        ),
    )
    config.update(
        architectures=["DSparkTargetKVDraftModel"],
        input_mode="target_kv",
        target_kv_contract=msgspec.to_builtins(contract),
        num_hidden_layers=2,
        num_target_layers=config["num_hidden_layers"],
        block_size=3,
        mask_token_id=vocab_size - 1,
        markov_rank=8,
        markov_head_type="vanilla",
        enable_confidence_head=False,
        test_observation_path=str(destination / "observations.jsonl"),
    )
    (destination / "config.json").write_text(json.dumps(config))
    weights = {}
    for path in root.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():  # noqa: SIM118 - safe_open is not iterable
                if name == "model.norm.weight" or name.startswith(
                    ("model.layers.0.", "model.layers.1.")
                ):
                    weights[name.removeprefix("model.")] = source.get_tensor(name)
    generator = torch.Generator().manual_seed(1729)
    weights["kv_encoder.projection.weight"] = (
        torch.randn(hidden_size, contract.feature_size, generator=generator)
        / contract.feature_size**0.5
    ).bfloat16()
    weights["kv_encoder.norm_weight"] = torch.ones(hidden_size, dtype=torch.bfloat16)
    for name in ("markov_w1", "markov_w2"):
        weights[f"markov_head.{name}.weight"] = (
            torch.randn(vocab_size, 8, generator=generator) * 0.01
        ).bfloat16()
    # Force a known proposal so a biased target request exercises full accept,
    # while ordinary target requests exercise rejection with the same weights.
    weights["markov_head.markov_w1.weight"][:, 0] = 1
    weights["markov_head.markov_w2.weight"][:, 0] = 0
    weights["markov_head.markov_w2.weight"][100, 0] = 10000
    save_file(weights, str(destination / "model.safetensors"))
    return contract


def exercise_target_kv_draft(
    test, *, model_path, directory, samples, responses, cuda_graph, enable_overlap=False
):
    destination = Path(directory) / f"synthetic-kv-draft-{cuda_graph}-{enable_overlap}"
    export_synthetic_kv_draft(model_path, destination, *samples[1])
    first_publication = len(test.catalog.publications)
    results = []
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    launch = test_utils._launch_server_process

    def observed_server(command, *args):
        return launch(
            [sys.executable, "-m", "sglang.test.dspark_target_kv_server"] + command[2:],
            *args,
        )

    def wait_admission(count=1):
        deadline = time.monotonic() + 20
        while True:
            state = requests.get(url + "/server_info", timeout=10).json()[
                "internal_states"
            ][0]["training_capture"]
            if state["states"].get("available", 0) >= count:
                return
            test.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.05)

    with (
        envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
        envs.SGLANG_TEST_RETRACT.override(False),
        patch.object(test_utils, "_launch_server_process", observed_server),
    ):
        server = test_utils.popen_launch_server(
            model_path,
            url,
            timeout=240,
            other_args=[
                "--training-capture-config",
                str(test.capture_path),
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                str(destination),
                *([] if enable_overlap else ["--disable-overlap-schedule"]),
                "--skip-server-warmup",
                "--enable-metrics",
                "--attention-backend",
                "triton",
                "--speculative-draft-attention-backend",
                "triton",
                "--mem-fraction-static",
                "0.25",
                "--max-total-tokens",
                "512",
                "--schedule-conservativeness",
                "0.05",
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
    try:
        for endpoint, payload in (
            ("update_weights_from_disk", {"model_path": model_path}),
            ("release_memory_occupation", {"tags": ["kv_cache"]}),
            ("resume_memory_occupation", {"tags": ["kv_cache"]}),
        ):
            rejection = requests.post(url + "/" + endpoint, json=payload, timeout=20)
            test.assertEqual(rejection.status_code, 400, rejection.text)
            test.assertIn("target-KV DSpark", rejection.text)
        for index, ((manifest, tensors), baseline) in enumerate(
            zip(samples, responses, strict=True)
        ):
            wait_admission()
            params = {
                "temperature": 0,
                "max_new_tokens": manifest.sequence.response_length,
                "ignore_eos": True,
            }
            if index == 2:
                params["logit_bias"] = {"100": 100.0}
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": tensors["token_ids"][
                        : manifest.sequence.prompt_length
                    ].tolist(),
                    "sampling_params": params,
                },
                timeout=120,
            )
            test.assertEqual(response.status_code, 200, response.text)
            actual = response.json()
            results.append(actual)
            test.assertEqual(actual["output_ids"], baseline["output_ids"])
            if index:
                test.assertGreater(actual["meta_info"]["cached_tokens"], 0)
        batch_indices = [1, 2, 1] if enable_overlap else [1, 2]
        wait_admission(len(batch_indices))
        batched = requests.post(
            url + "/generate",
            json={
                "input_ids": [
                    samples[index][1]["token_ids"][
                        : samples[index][0].sequence.prompt_length
                    ].tolist()
                    for index in batch_indices
                ],
                "sampling_params": [
                    {
                        "temperature": 0,
                        "max_new_tokens": len(responses[index]["output_ids"]),
                        "ignore_eos": True,
                        **({"logit_bias": {"100": 100.0}} if index == 2 else {}),
                    }
                    for index in batch_indices
                ],
            },
            timeout=120,
        )
        test.assertEqual(batched.status_code, 200, batched.text)
        for actual, index in zip(batched.json(), batch_indices, strict=True):
            results.append(actual)
            test.assertEqual(actual["output_ids"], responses[index]["output_ids"])
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        extra_cases = (
            (
                "sampling_penalty",
                {
                    "temperature": 0.8,
                    "top_k": 16,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                    "frequency_penalty": 0.15,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
            ),
            (
                "stop_token",
                {
                    "temperature": 0,
                    "min_new_tokens": 1,
                    "max_new_tokens": 8,
                    "stop_token_ids": [100],
                    "logit_bias": {"100": 100.0},
                },
            ),
            (
                "eos",
                {
                    "temperature": 0,
                    "min_new_tokens": 1,
                    "max_new_tokens": 8,
                    "logit_bias": {str(tokenizer.eos_token_id): 100.0},
                },
            ),
            (
                "grammar",
                {"temperature": 0.8, "max_new_tokens": 16, "regex": "[0-9]{12}"},
            ),
        )
        for name, params in extra_cases:
            wait_admission()
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": samples[0][1]["token_ids"][
                        : samples[0][0].sequence.prompt_length
                    ].tolist(),
                    "sampling_params": params,
                },
                timeout=120,
            )
            test.assertEqual(response.status_code, 200, response.text)
            actual = response.json()
            results.append(actual)
            if name in ("stop_token", "eos"):
                test.assertEqual(len(actual["output_ids"]), 2, actual)
                test.assertEqual(
                    actual["output_ids"][-1],
                    100 if name == "stop_token" else tokenizer.eos_token_id,
                )
                test.assertEqual(actual["meta_info"]["finish_reason"]["type"], "stop")
            elif name == "grammar":
                test.assertRegex(actual["text"], r"^[0-9]{12}$")
        if enable_overlap:
            wait_admission()
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": [100, 200, 300, 400] * 63,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 4,
                        "ignore_eos": True,
                        "logit_bias": {"100": 100.0},
                    },
                },
                timeout=120,
            )
            test.assertEqual(response.status_code, 200, response.text)
            actual = response.json()
            test.assertEqual(actual["output_ids"], [100] * 4)
            results.append(actual)
        publications = test.catalog.wait_publications(
            first_publication + len(results), timeout=30
        )[first_publication:]
        references = [
            torch.load(path, weights_only=True)
            for path in sorted((destination / "capture-reference").glob("*.pt"))
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
            check_capture_snapshot(test, manifest, tensors, references)
        if enable_overlap:
            test.assertTrue(any(row["result_lag"] > 0 for row in references))
            test.assertTrue(any(row["batch_size"] == 3 for row in references))
            if cuda_graph:
                test.assertTrue(
                    any(
                        row["batch_size"] == 3 and row["cuda_graph"]
                        for row in references
                    )
                )
            test.assertTrue(
                any(manifest.sequence.total_length == 256 for manifest, _ in captured)
            )
        test.assertIn(
            "eos", [manifest.sequence.stop_reason for manifest, _ in captured]
        )
        test.assertIn(
            "stop_token", [manifest.sequence.stop_reason for manifest, _ in captured]
        )
        capture_state = requests.get(url + "/server_info", timeout=10).json()[
            "internal_states"
        ][0]["training_capture"]
        test.assertEqual(capture_state["enable_overlap"], enable_overlap)
        test.assertEqual(
            capture_state["counters"].get("admitted", 0), len(results), capture_state
        )
        test.assertGreater(
            capture_state["counters"].get("speculative_verify_forwards", 0),
            0,
            capture_state,
        )
        test.assertGreater(
            capture_state["counters"].get("speculative_commits_copied", 0),
            0,
            capture_state,
        )
        if cuda_graph:
            test.assertGreater(
                capture_state["counters"].get("cuda_graph_forwards", 0),
                0,
                capture_state,
            )
        if enable_overlap:
            wait_admission()
            capture_state = exercise_capture_abort(
                test,
                url=url,
                rid=f"abort-dspark-overlap-{cuda_graph}",
                prompt=samples[0][1]["token_ids"][:8].tolist(),
                max_new_tokens=248,
            )
            test.assertEqual(
                len(test.catalog.publications), first_publication + len(results)
            )
        print(
            json.dumps(
                {
                    "speculative_capture": capture_state,
                    "samples": len(captured),
                    "cuda_graph_enabled": cuda_graph,
                    "overlap_enabled": enable_overlap,
                    "aborted": enable_overlap,
                    "padded_graph": enable_overlap and cuda_graph,
                }
            ),
            flush=True,
        )
        observations = [
            json.loads(line)
            for line in (destination / "observations.jsonl").read_text().splitlines()
        ]
        test.assertTrue(any(row["kind"] == "projection" for row in observations))
        commits = [row for row in observations if row["kind"] == "verify"]
        test.assertTrue(commits)
        test.assertTrue(any(row["num_reject"] > 0 for row in commits))
        test.assertTrue(any(max(row["num_commit"]) > 1 for row in commits))
        test.assertTrue(any(len(row["num_commit"]) > 1 for row in commits))
        verifies = [row for row in observations if row["kind"] == "target_verify"]
        test.assertTrue(verifies)
        test.assertEqual(any(row["cuda_graph"] for row in verifies), cuda_graph)
        print(
            json.dumps(
                {"target_kv_serving": observations, "cuda_graph_enabled": cuda_graph}
            ),
            flush=True,
        )
        from sglang.test.dspark_capture_pressure import exercise_dspark_capture_pressure

        publications += exercise_dspark_capture_pressure(
            test,
            url=url,
            directory=destination,
            cuda_graph=cuda_graph,
            enable_overlap=enable_overlap,
        )
    finally:
        kill_process_tree(server.pid)
        server.wait(timeout=20)
    for publication in publications:
        test.read_sample(publication)

"""Synthetic KV-draft export and live serving checks, never a trained artifact."""

import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import msgspec
import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file

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
    test, *, model_path, directory, samples, responses, cuda_graph
):
    destination = Path(directory) / f"synthetic-kv-draft-{cuda_graph}"
    export_synthetic_kv_draft(model_path, destination, *samples[1])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    launch = test_utils._launch_server_process

    def observed_server(command, *args):
        return launch(
            [sys.executable, "-m", "sglang.test.dspark_target_kv_server"] + command[2:],
            *args,
        )

    with (
        patch.dict("os.environ", {"SGLANG_RAGGED_VERIFY_MODE": "static"}),
        patch.object(test_utils, "_launch_server_process", observed_server),
    ):
        server = test_utils.popen_launch_server(
            model_path,
            url,
            timeout=240,
            other_args=[
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                str(destination),
                "--disable-overlap-schedule",
                "--skip-server-warmup",
                "--skip-tokenizer-init",
                "--attention-backend",
                "triton",
                "--speculative-draft-attention-backend",
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
            test.assertEqual(actual["output_ids"], baseline["output_ids"])
            if index:
                test.assertGreater(actual["meta_info"]["cached_tokens"], 0)
        batched = requests.post(
            url + "/generate",
            json={
                "input_ids": [
                    sample[1]["token_ids"][: sample[0].sequence.prompt_length].tolist()
                    for sample in samples[1:]
                ],
                "sampling_params": [
                    {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True},
                    {
                        "temperature": 0,
                        "max_new_tokens": 3,
                        "ignore_eos": True,
                        "logit_bias": {"100": 100.0},
                    },
                ],
            },
            timeout=120,
        )
        test.assertEqual(batched.status_code, 200, batched.text)
        for actual, baseline in zip(batched.json(), responses[1:], strict=True):
            test.assertEqual(actual["output_ids"], baseline["output_ids"])
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
    finally:
        kill_process_tree(server.pid)
        server.wait(timeout=20)

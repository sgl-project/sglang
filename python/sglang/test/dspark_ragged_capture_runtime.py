"""Real confidence-scheduled verify -> accepted training samples -> Store."""

import hashlib
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_sps import SpsCostTable
from sglang.srt.utils import kill_process_tree
from sglang.test import test_utils
from sglang.test.dspark_capture_observer import check_capture_snapshot
from sglang.test.training_capture_utils import read_snapshot


def export_confidence_draft(model_path, destination):
    """Synthetic test weights exercise scheduling; they are not a trained draft."""
    destination.mkdir()
    root = Path(model_path)
    config = json.loads((root / "config.json").read_text())
    hidden, vocab = config["hidden_size"], config["vocab_size"]
    config.update(
        architectures=["DSparkDraftModel"],
        num_target_layers=config["num_hidden_layers"],
        num_hidden_layers=2,
        block_size=3,
        mask_token_id=vocab - 1,
        target_layer_ids=[0, 14, 26],
        markov_rank=8,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=False,
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
    weights["fc.weight"] = (
        torch.randn(hidden, 3 * hidden, generator=generator) / (3 * hidden) ** 0.5
    ).bfloat16()
    weights["hidden_norm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
    weights["markov_head.markov_w1.weight"] = torch.zeros(
        vocab, 8, dtype=torch.bfloat16
    )
    weights["markov_head.markov_w2.weight"] = torch.zeros(
        vocab, 8, dtype=torch.bfloat16
    )
    weights["markov_head.markov_w1.weight"][:, 0] = 1
    weights["markov_head.markov_w2.weight"][100, 0] = 10000
    weights["confidence_head.proj.weight"] = torch.zeros(1, hidden)
    weights["confidence_head.proj.bias"] = torch.zeros(1)
    save_file(weights, str(destination / "model.safetensors"))
    table = SpsCostTable(
        sample_batch_tokens=[1, 16],
        sample_steps_per_sec=[1000.0, 500.0],
        max_batch_tokens=16,
    )
    (destination / "sps.json").write_text(table.to_json())


def exercise_ragged_capture(test, *, prompt, baseline, mode, cuda_graph, tp_size=1):
    # Triton falls back to eager for ragged target verification.
    attention_backend = "fa3" if mode == "compact" else "triton"
    destination = Path(test.temporary.name) / f"ragged-tp{tp_size}-{mode}-{cuda_graph}"
    export_confidence_draft(test.model_path, destination)
    config = json.loads(test.capture_path.read_text())
    config["journal_directory"] = str(destination / "journal")
    capture_path = destination / "capture.json"
    capture_path.write_text(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    launch = test_utils._launch_server_process

    def observed_server(command, *args):
        return launch(
            [sys.executable, "-m", "sglang.test.dspark_capture_server"] + command[2:],
            *args,
        )

    with (
        patch.object(test_utils, "_launch_server_process", observed_server),
        envs.SGLANG_RAGGED_VERIFY_MODE.override(mode),
        envs.SGLANG_PREP_IN_CUDA_GRAPH.override(True),
    ):
        server = test_utils.popen_launch_server(
            test.model_path,
            url,
            timeout=240,
            other_args=[
                "--tp-size",
                str(tp_size),
                "--training-capture-config",
                str(capture_path),
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                str(destination),
                "--speculative-dspark-sps-table-path",
                str(destination / "sps.json"),
                "--speculative-draft-attention-backend",
                "triton",
                "--attention-backend",
                attention_backend,
                "--skip-server-warmup",
                "--skip-tokenizer-init",
                "--mem-fraction-static",
                "0.25",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "4",
                "--chunked-prefill-size",
                "128",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--cuda-graph-backend-decode",
                "full" if cuda_graph else "disabled",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "4",
                *([] if cuda_graph else ["--disable-overlap-schedule"]),
            ],
        )
    first_publication = len(test.catalog.publications)
    responses = {}
    try:
        setting = requests.post(
            url + "/set_internal_state",
            json={"server_args": {"dspark_force_budget_frac": 0.625}},
            timeout=20,
        )
        test.assertEqual(setting.status_code, 200, setting.text)
        updates = setting.json()
        updates = updates if isinstance(updates, list) else [updates]
        test.assertTrue(
            all(
                item.get("updated") if isinstance(item, dict) else item is True
                for item in updates
            ),
            updates,
        )
        for batched in (False, True):
            params = [
                {"temperature": 0, "max_new_tokens": len(baseline), "ignore_eos": True},
                {
                    "temperature": 0,
                    "max_new_tokens": 9,
                    "ignore_eos": True,
                    "logit_bias": {"100": 100.0},
                },
            ]
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": [prompt, prompt] if batched else prompt,
                    "sampling_params": params if batched else params[0],
                },
                timeout=120,
            )
            test.assertEqual(response.status_code, 200, response.text)
            actual = response.json() if batched else [response.json()]
            expected = [baseline, [100] * 9] if batched else [baseline]
            for result, tokens in zip(actual, expected, strict=True):
                test.assertEqual(result["output_ids"], tokens)
                responses[
                    hashlib.sha256(result["meta_info"]["id"].encode()).hexdigest()
                ] = result
            test.catalog.wait_publications(
                first_publication + len(responses), timeout=45
            )
        state_response = requests.get(url + "/server_info", timeout=10)
        state_response.raise_for_status()
        state = state_response.json()["internal_states"][0]["training_capture"]
        test.assertEqual(state["enable_overlap"], cuda_graph)
    finally:
        kill_process_tree(server.pid)
        server.wait(timeout=20)
    references = [
        torch.load(path, weights_only=True)
        for path in sorted((destination / "capture-reference").rglob("*.pt"))
    ]
    test.assertEqual({item["tp_rank"] for item in references}, set(range(tp_size)))
    for publication in test.catalog.wait_publications(
        first_publication + len(responses)
    )[first_publication:]:
        manifest, tensors = read_snapshot(test.reader, publication)
        test.assertEqual(manifest.topology.tp_size, tp_size)
        result = responses.pop(manifest.provenance.trace_id)
        test.assertEqual(tensors["token_ids"].tolist(), prompt + result["output_ids"])
        check_capture_snapshot(test, manifest, tensors, references)
    test.assertFalse(responses)
    acceptance = [
        json.loads(line)
        for path in sorted(destination.glob("acceptance-tp*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    test.assertEqual({item["tp_rank"] for item in acceptance}, set(range(tp_size)))
    test.assertTrue(
        any(
            item["verify_lens"] and len(set(item["verify_lens"])) > 1
            for item in acceptance
        ),
        acceptance,
    )
    test.assertTrue(any(any(item["cap_trim_lens"]) for item in acceptance), acceptance)
    if mode == "compact":
        test.assertTrue(
            any(
                item["verify_count"] and item["verify_count"] < item["verify_width"]
                for item in references
            )
        )
        if cuda_graph:
            test.assertTrue(any(item["verify_padding"] > 0 for item in references))
            test.assertTrue(any(item["folded"] for item in acceptance), acceptance)
    if cuda_graph:
        test.assertTrue(
            any(
                item["cuda_graph"] and item["verify_count"] is not None
                for item in references
            )
        )
    print(
        json.dumps(
            {
                "ragged_capture": mode,
                "graph_overlap": cuda_graph,
                "tp_size": tp_size,
                "attention_backend": attention_backend,
                "state": state,
                "accept_layouts": acceptance,
            }
        ),
        flush=True,
    )

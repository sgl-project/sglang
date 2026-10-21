# SPDX-License-Identifier: Apache-2.0
"""Launch an owned one-GPU server and exercise control with actual generation.

Select one GPU using CUDA_VISIBLE_DEVICES. A local OPT config and --load-format
dummy suffice for an interface smoke test; this does not evaluate model quality.
--hicache also checks independent MOVE/COPY and subsequent loaded-prefix reuse.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--hicache", action="store_true")
    parser.add_argument(
        "--quota",
        action="store_true",
        help="Verify configured task guidance in the real scheduler",
    )
    parser.add_argument(
        "--retention",
        action="store_true",
        help="Verify configured policy children and explicit priority precedence",
    )
    parser.add_argument(
        "--storage",
        action="store_true",
        help="Round-trip KV through file storage across a physical cache flush",
    )
    parser.add_argument(
        "--check-weight-update",
        action="store_true",
        help="OPT fixture: mutate attention LayerNorm biases and verify a cold new namespace",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.storage or args.retention:
        args.hicache = True
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output_dir or tempfile.mkdtemp(prefix="sglang-kv-server-smoke-"))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "server.log").exists() or (output / "result.json").exists():
        raise ValueError(
            "Use a fresh output directory to retain earlier validation artifacts"
        )
    identity = uuid.uuid4().hex
    engine_id = "smoke-" + identity
    namespace = "smoke-weights-" + identity
    control_config = {"namespace": namespace, "engine_id": engine_id}
    if args.retention:
        control_config["policies"] = [
            {"name": "retain", "options": {"priority": 50, "ttl_seconds": 60}}
        ]
    if args.quota:
        control_config.setdefault("policies", []).append(
            {"name": "quota", "implementation": "quota"}
        )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--load-format",
        args.load_format,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--skip-tokenizer-init",
        "--cuda-graph-backend-decode",
        "disabled",
        "--disable-overlap-schedule",
        "--attention-backend",
        "torch_native",
        "--sampling-backend",
        "pytorch",
        "--grammar-backend",
        "none",
        "--mem-fraction-static",
        "0.10",
        "--max-total-tokens",
        "128",
        "--context-length",
        "128",
        "--max-running-requests",
        "2",
        "--dtype",
        "float16",
        "--kv-control-config",
        json.dumps(control_config),
    ]
    if args.hicache:
        command += [
            "--enable-hierarchical-cache",
            "--hicache-ratio",
            "2",
            "--hicache-write-policy",
            "write_back",
            "--hicache-mem-layout",
            "layer_first",
            "--hicache-io-backend",
            "direct",
        ]
    if args.storage:
        command += [
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-backend-extra-config",
            json.dumps({"file_storage_path": str(output / "cold")}),
        ]
    environment = dict(
        os.environ,
        PYTHONPATH=str(root / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
    )
    base = f"http://127.0.0.1:{port}"
    with (output / "server.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 360
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Server exited {proc.returncode}; see {output / 'server.log'}"
                    )
                try:
                    if requests.get(base + "/health", timeout=15).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("Server startup exceeded six minutes")

            def get_capabilities():
                response = requests.get(base + "/v1/kv/capabilities", timeout=15)
                response.raise_for_status()
                return response.json()["data"]

            def control(body):
                response = requests.post(base + "/v1/kv/control", json=body, timeout=15)
                response.raise_for_status()
                result = response.json()
                assert result["success"], result
                return result["data"]

            owner = get_capabilities()
            assert (
                owner["engine_id"] == engine_id and owner["namespace"] == namespace
            ), owner
            if args.hicache:
                assert (
                    owner["executor"]["directions"]["gpu_copy_approval"]
                    == "scheduler_prefill_budget"
                ), owner
            context = {k: owner[k] for k in ("engine_id", "engine_epoch", "namespace")}
            context["caller_id"] = "request"
            hints = {
                "protocol_version": "0.1",
                "message_id": "attached-smoke",
                "actions": [
                    {
                        "action_id": "keep",
                        "action_type": "sglang.set_priority",
                        "action_version": "1.0",
                        "payload": {
                            "target": "request_cached",
                            "location": {"tier": "gpu"},
                            "claim_id": "smoke",
                            "revision": 1,
                            "value": 90,
                            "ttl_seconds": 60,
                        },
                    }
                ],
            }
            body = {
                "input_ids": [17, 23, 29, 31],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 2,
                    "ignore_eos": True,
                },
                "kv_hints": hints,
            }
            if args.quota:
                registration = control(
                    {
                        "kind": "submit",
                        "context": dict(context, caller_id="router"),
                        "envelope": {
                            "protocol_version": "0.1",
                            "message_id": "quota-associate",
                            "actions": [
                                {
                                    "action_id": "associate",
                                    "action_type": "sglang.associate",
                                    "action_version": "1.0",
                                    "payload": {
                                        "session_id": "smoke-task",
                                        "continuation_id": "prompt",
                                        "expected_revision": 0,
                                        "target": {
                                            "kind": "tokens",
                                            "namespace": namespace,
                                            "token_ids": body["input_ids"],
                                        },
                                    },
                                }
                            ],
                        },
                    }
                )
                assert registration["results"][0]["update"]["status"] == "completed", (
                    registration
                )
                hints["actions"].append(
                    {
                        "action_id": "quota-bind",
                        "action_type": "sglang.signal",
                        "action_version": "1.0",
                        "payload": {
                            "target": {
                                "kind": "session",
                                "namespace": namespace,
                                "session_id": "smoke-task",
                            },
                            "signal": "resume",
                        },
                    }
                )
            generated = requests.post(base + "/generate", json=body, timeout=60)
            generated.raise_for_status()
            generated = generated.json()
            messages = control(
                {"kind": "lookup", "context": context, "message_id": "attached-smoke"}
            )
            results = [
                r
                for message in messages
                for r in message["results"]
                if r["ref"]["action_id"] == "keep"
            ]
            quota_binding = [
                r
                for message in messages
                for r in message["results"]
                if r["ref"]["action_id"] == "quota-bind"
            ]
            if args.quota:
                assert (
                    len(quota_binding) == 1
                    and quota_binding[0]["update"]["status"] == "completed"
                ), quota_binding
            assert (
                len(results) == 1 and results[0]["update"]["status"] == "completed"
            ), results
            ref = results[0]["ref"]
            movements = []
            policy_validation = []
            if args.hicache:
                end = max(
                    o["coverage"]["end_token"]
                    for o in results[0]["update"]["target_outcomes"]
                )
                tokens = (body["input_ids"] + generated["output_ids"])[:end]
                target = {
                    "kind": "tokens",
                    "namespace": owner["namespace"],
                    "token_ids": tokens,
                }
                router = dict(context, caller_id="router")

                def movement(kind, src, dst, message):
                    receipt = control(
                        {
                            "kind": "submit",
                            "context": router,
                            "envelope": {
                                "protocol_version": "0.1",
                                "message_id": message,
                                "actions": [
                                    {
                                        "action_id": "a",
                                        "action_type": "sglang." + kind,
                                        "action_version": "1.0",
                                        "payload": {
                                            "target": target,
                                            "src": {"tier": src},
                                            "dst": {"tier": dst},
                                        },
                                    }
                                ],
                            },
                        }
                    )
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        result = receipt["results"][0]
                        status = result["update"]["status"]
                        if status not in (
                            "received",
                            "queued",
                            "accepted",
                            "running",
                            "cancelling",
                        ):
                            assert status == "completed", result
                            return result
                        time.sleep(0.01)
                        receipt = control(
                            {"kind": "query", "context": router, "message_id": message}
                        )
                    raise TimeoutError("Control operation did not retire")

                def retention(message, importance, expected):
                    receipt = control(
                        {
                            "kind": "submit",
                            "context": router,
                            "envelope": {
                                "protocol_version": "0.1",
                                "message_id": message,
                                "actions": [
                                    {
                                        "action_id": "signal",
                                        "action_type": "sglang.signal",
                                        "action_version": "1.0",
                                        "payload": {
                                            "target": target,
                                            "signal": "importance",
                                            "importance": importance,
                                        },
                                    }
                                ],
                            },
                        }
                    )
                    parent = receipt["results"][0]
                    assert parent["update"]["status"] == "completed", parent
                    (child,) = parent["update"]["derived_operations"]
                    result = control(
                        {
                            "kind": "query",
                            "context": child["context"],
                            "message_id": child["message_id"],
                        }
                    )
                    assert result["results"][0]["update"]["status"] == "completed", (
                        result
                    )
                    resident = control(
                        {"kind": "residency", "context": router, "target": target}
                    )
                    assert resident["items"] and all(
                        row["effective_priority"] == expected
                        for row in resident["items"]
                    ), resident
                    policy_validation.append(
                        {
                            "signal": parent,
                            "children": result,
                            "effective_priority": expected,
                        }
                    )

                if args.retention:
                    assert "retain" in owner["policies"]
                    retention("retain-before-move", 200, 90)
                movements.append(
                    movement(
                        "move_to", "gpu", "storage" if args.storage else "host", "save"
                    )
                )
                if args.storage:
                    flushed = requests.post(base + "/flush_cache", timeout=15)
                    flushed.raise_for_status()
                    before_epoch = owner["engine_epoch"]
                    owner = get_capabilities()
                    assert owner["engine_epoch"] != before_epoch
                    router = dict(router, engine_epoch=owner["engine_epoch"])
                absent = control(
                    {"kind": "residency", "context": router, "target": target}
                )
                assert not absent["items"] and absent["missing"], absent
                host = control(
                    {
                        "kind": "residency",
                        "context": router,
                        "target": target,
                        "location": {"tier": "host"},
                    }
                )
                if args.storage:
                    assert not host["items"] and host["missing"], host
                else:
                    assert host["items"] and not host["missing"], host
                movements.append(
                    movement(
                        "copy_to",
                        "storage" if args.storage else "host",
                        "gpu",
                        "restore",
                    )
                )
                loaded = control(
                    {"kind": "residency", "context": router, "target": target}
                )
                assert loaded["items"] and not loaded["missing"], loaded
                assert all(i["effective_priority"] == 0 for i in loaded["items"]), (
                    loaded
                )
                if args.retention:
                    retention("retain-after-load", 55, 55)
            repeated = requests.post(
                base + "/generate",
                json={k: v for k, v in body.items() if k != "kv_hints"},
                timeout=60,
            )
            repeated.raise_for_status()
            repeated = repeated.json()
            assert generated["output_ids"] == repeated["output_ids"], (
                generated,
                repeated,
            )
            assert (
                repeated["meta_info"]["cached_tokens"] >= len(body["input_ids"]) - 1
            ), repeated
            weights_validation = None
            if args.check_weight_update:

                def checker(action, **extra):
                    response = requests.post(
                        base + "/weights_checker",
                        json={"action": action, "selector": "target", **extra},
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
                    assert data["success"], data
                    return data

                before_checksum = checker("checksum")["per_engine_checksum"]
                old_owner = owner
                # Change only OPT attention LayerNorm biases, keeping the
                # fixture's large projection matrices numerically bounded.
                skipped = [
                    "embed_tokens",
                    "embed_positions",
                    "lm_head",
                    ".self_attn.",
                    ".fc1.",
                    ".fc2.",
                    "final_layer_norm",
                    "self_attn_layer_norm.weight",
                ]
                checker("reset_tensors", skip_tensor_list=skipped)
                owner = get_capabilities()
                assert owner["namespace"] != old_owner["namespace"]
                assert owner["engine_epoch"] != old_owner["engine_epoch"]
                assert not owner["updating_weights"]
                after_checksum = checker("checksum")["per_engine_checksum"]
                assert before_checksum != after_checksum
                cold = requests.post(
                    base + "/generate",
                    json={k: v for k, v in body.items() if k != "kv_hints"},
                    timeout=60,
                )
                cold.raise_for_status()
                cold = cold.json()
                assert cold["meta_info"]["cached_tokens"] == 0, cold
                warm = requests.post(
                    base + "/generate",
                    json={k: v for k, v in body.items() if k != "kv_hints"},
                    timeout=60,
                )
                warm.raise_for_status()
                warm = warm.json()
                assert cold["output_ids"] == warm["output_ids"]
                assert warm["meta_info"]["cached_tokens"] >= len(body["input_ids"]) - 1
                weights_validation = dict(
                    before_namespace=old_owner["namespace"],
                    after_namespace=owner["namespace"],
                    before_checksum=before_checksum,
                    after_checksum=after_checksum,
                    cold=cold,
                    warm=warm,
                    skip_tensor_list=skipped,
                )
            flush = requests.post(base + "/flush_cache", timeout=15)
            flush.raise_for_status()
            after = get_capabilities()
            assert after["engine_epoch"] != owner["engine_epoch"]
            if args.hicache:
                assert (
                    after["executor"]["directions"]["gpu_copy_approval"]
                    == "scheduler_prefill_budget"
                ), after
            stale = requests.post(
                base + "/v1/kv/control",
                json={
                    "kind": "query",
                    "context": ref["context"],
                    "message_id": "attached-smoke",
                },
                timeout=15,
            )
            assert stale.status_code == 409, stale.text
            result = {
                "passed": True,
                "model_path": args.model_path,
                "load_format": args.load_format,
                "hicache": args.hicache,
                "storage": args.storage,
                "capabilities": owner,
                "generated": generated,
                "attached_result": results[0],
                "movements": movements,
                "repeat": repeated,
                "flush_epoch_rotated": True,
                "stale_status": stale.status_code,
                "weights_validation": weights_validation,
                "policy_validation": policy_validation,
                "quota_binding": quota_binding,
            }
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "passed": True,
                        "hicache": args.hicache,
                        "cached_tokens": repeated["meta_info"]["cached_tokens"],
                        "artifacts": str(output),
                    }
                ),
                flush=True,
            )
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    main()

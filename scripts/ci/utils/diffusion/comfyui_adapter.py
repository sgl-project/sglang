# SPDX-License-Identifier: Apache-2.0
"""Drive a headless ComfyUI server for the cross-framework comparison.

ComfyUI is a workflow engine, not an OpenAI-compatible server: a run is a graph
POSTed to ``/prompt`` and collected from ``/history``. Two things that make a
comparison against it meaningful, and that this module exists to enforce:

*Weight parity.* ComfyUI's shipped MiniMax-H3 path is pruned + int8 with an
NVFP4 text encoder, which is not the lossless path SGLang runs. A case must
point at the unpruned bf16 repack and pass ``--bf16-text-enc --bf16-unet``,
or ComfyUI also casts the text encoder to fp16 on its own.

*Node schemas are not guessable.* Input names and enum values move between
ComfyUI releases, and a wrong value is accepted as a plain validation error
long after the server booted. Every workflow is checked against the live
``/object_info`` before it is submitted, and a mismatch reports the
authoritative schema instead of failing at request time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from string import Template
from typing import Any

import requests

WORKFLOWS_DIR = Path(__file__).parent / "comfyui_workflows"
HEALTH_ENDPOINT = "/system_stats"
POLL_INTERVAL_S = 1.0


def build_launch_cmd(fw_cfg: dict, port: int, host: str) -> list[str]:
    """ComfyUI runs from its own checkout and virtualenv, not from ours.

    Those paths are per-machine, so the config carries ``$VAR`` references
    (``COMFY_ROOT``, ``COMFY_PYTHON``) that expand from the environment and
    keep the committed case machine-agnostic.
    """
    comfy_root = os.path.expandvars(fw_cfg.get("comfy_root", ""))
    if not comfy_root or "$" in comfy_root:
        raise ValueError(
            "comfyui needs 'comfy_root' (path to the ComfyUI checkout); got "
            f"{fw_cfg.get('comfy_root')!r} — set COMFY_ROOT in the environment"
        )
    python_bin = os.path.expandvars(fw_cfg.get("python_bin", "python3"))
    if "$" in python_bin:
        raise ValueError(
            f"comfyui 'python_bin' unresolved: {fw_cfg.get('python_bin')!r} — "
            "set COMFY_PYTHON in the environment"
        )
    cmd = [
        python_bin,
        str(Path(comfy_root) / "main.py"),
        "--listen",
        host,
        "--port",
        str(port),
    ]
    serve_args = fw_cfg.get("serve_args", "").strip()
    if serve_args:
        cmd += serve_args.split()
    return cmd


def _workflow_path(case: dict) -> Path:
    """Workflows are per case: the graph differs by model, not just by size."""
    name = case.get("comfyui_workflow") or f"{case['id']}.json"
    path = WORKFLOWS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"No ComfyUI workflow for case {case['id']!r} at {path}. "
            "Each case needs its own API-format graph; export one from the UI "
            "or hand-write it, then validate with --validate-comfyui-workflow."
        )
    return path


def render_workflow(case: dict, fw_cfg: dict, config: dict) -> dict:
    """Fill the case's parameters into its workflow template.

    Substitution is explicit rather than structural so that the template stays
    a readable API-format graph that can be pasted into ComfyUI to reproduce a
    run by hand.
    """
    substitutions = {
        "prompt": case["prompt"],
        "negative_prompt": case.get("negative_prompt", ""),
        "seed": case.get("seed", 42),
        "steps": case.get("num_inference_steps", 20),
        "cfg": case.get("guidance_scale", 1.0),
        "width": case.get("width", 0),
        "height": case.get("height", 0),
        "num_frames": case.get("num_frames", 1),
        "fps": case.get("fps", 16),
        "image_path": config.get("comfyui_ref_image", ""),
    }
    substitutions.update(fw_cfg.get("workflow_vars", {}))

    raw = _workflow_path(case).read_text()
    try:
        rendered = Template(raw).substitute(substitutions)
    except KeyError as exc:
        raise KeyError(
            f"ComfyUI workflow for {case['id']!r} references ${exc.args[0]}, which "
            f"is neither a case field nor a framework workflow_var. Known keys: "
            f"{sorted(substitutions)}"
        ) from exc
    graph = json.loads(rendered)
    # Templates carry `_`-prefixed keys to document the graph; ComfyUI reads
    # every top-level key as a node and rejects them.
    return {k: v for k, v in graph.items() if not k.startswith("_")}


def validate_workflow(base_url: str, workflow: dict) -> None:
    """Check node classes and input names against the running server.

    Raises before submission so a schema drift reads as "this input moved",
    not as an opaque validation failure mid-benchmark.
    """
    resp = requests.get(f"{base_url}/object_info", timeout=30)
    resp.raise_for_status()
    schema = resp.json()

    problems: list[str] = []
    for node_id, node in workflow.items():
        class_type = node.get("class_type")
        if class_type not in schema:
            problems.append(
                f"node {node_id}: unknown class_type {class_type!r}; "
                f"this ComfyUI build does not provide it"
            )
            continue
        spec = schema[class_type].get("input", {})
        known = set(spec.get("required", {})) | set(spec.get("optional", {}))
        for input_name, value in (node.get("inputs") or {}).items():
            if input_name not in known:
                problems.append(
                    f"node {node_id} ({class_type}): unknown input {input_name!r}; "
                    f"accepted inputs are {sorted(known)}"
                )
                continue
            # Enum inputs are declared as a list of allowed values.
            declared = spec.get("required", {}).get(input_name) or spec.get(
                "optional", {}
            ).get(input_name)
            if (
                isinstance(declared, list)
                and declared
                and isinstance(declared[0], list)
                and not isinstance(value, list)
                and value not in declared[0]
            ):
                problems.append(
                    f"node {node_id} ({class_type}).{input_name}={value!r} is not an "
                    f"accepted value; allowed: {declared[0]}"
                )

    if problems:
        raise ValueError(
            "ComfyUI workflow does not match the server's schema:\n  "
            + "\n  ".join(problems)
        )


def submit_and_wait(base_url: str, workflow: dict, timeout: float) -> float:
    """Run one graph and return its wall-clock seconds.

    Timed the same way as the other frameworks: client-side, from submission
    to the point the outputs are retrievable.
    """
    start = time.time()
    resp = requests.post(
        f"{base_url}/prompt", json={"prompt": workflow}, timeout=timeout
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"ComfyUI rejected the workflow: {resp.status_code} {resp.text[:500]}"
        )
    prompt_id = resp.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {resp.text[:500]}")

    while True:
        history = requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
        history.raise_for_status()
        entry = history.json().get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(f"ComfyUI run failed: {json.dumps(status)[:800]}")
            if entry.get("outputs"):
                return time.time() - start
        if time.time() - start > timeout:
            raise TimeoutError(
                f"ComfyUI prompt {prompt_id} did not finish in {timeout}s"
            )
        time.sleep(POLL_INTERVAL_S)


def send_request(
    base_url: str, case: dict, fw_cfg: dict, config: dict, timeout: float
) -> float:
    workflow = render_workflow(case, fw_cfg, config)
    validate_workflow(base_url, workflow)
    return submit_and_wait(base_url, workflow, timeout)


def describe_parity(fw_cfg: dict) -> list[str]:
    """Flags this config is missing that silently change precision.

    Reported rather than auto-added: a case that wants a quantized ComfyUI
    path is legitimate, it just is not the lossless comparison.
    """
    serve_args = fw_cfg.get("serve_args", "")
    missing = [
        flag for flag in ("--bf16-text-enc", "--bf16-unet") if flag not in serve_args
    ]
    return missing

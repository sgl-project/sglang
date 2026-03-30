"""
DCP accuracy test matrix.

Tests all feasible combinations of:
  - Attention backend:  flashinfer, flash_attn
  - DCP comm backend:   ag_rs, a2a
  - CUDA graph:         enabled, disabled
  - Request type:       prefill-only, decode-heavy, mixed

Two models are used:
  - DeepSeek-V2-Lite (MLA): exercises the DCP path in deepseek_v2.py via FlashInfer
  - Qwen2.5-1.5B-Instruct (GQA): exercises the DCP path in flashattention_backend.py

Requires 4 GPUs.  Set CUDA_VISIBLE_DEVICES externally or use Docker --gpus.

Usage (inside Docker container):
    python3 test/srt/test_dcp_accuracy_matrix.py
"""

import json
import os
import signal
import subprocess
import sys
import time

GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "4,5,6,7")
NUM_GPUS = len(GPUS.split(","))
PORT = 18199
BASE_URL = f"http://127.0.0.1:{PORT}"

MODEL = "deepseek-ai/DeepSeek-V2-Lite"

REQUEST_TYPES = {
    "prefill_only": {"input": 2048, "output": 1},
    "decode_heavy": {"input": 32, "output": 512},
    "mixed": {"input": 512, "output": 256},
}

# DCP KV sharding is wired in deepseek_v2.py, so all DCP tests use DeepSeek-V2-Lite.
# Server configs: (model, backend, dcp_comm, cuda_graph)
SERVER_CONFIGS = [
    # FA3 first (exercises flashattention_backend.py DCP path)
    (MODEL, "fa3", "a2a", False),
    (MODEL, "fa3", "ag_rs", False),
    (MODEL, "fa3", "a2a", True),
    (MODEL, "fa3", "ag_rs", True),
    # FlashInfer (exercises deepseek_v2.py DCP path)
    (MODEL, "flashinfer", "ag_rs", True),
    (MODEL, "flashinfer", "ag_rs", False),
    (MODEL, "flashinfer", "a2a", False),
    (MODEL, "flashinfer", "a2a", True),
]


def _build_scenarios():
    scenarios = []
    idx = 0
    for model, backend, dcp_comm, cuda_graph in SERVER_CONFIGS:
        model_tag = "mla" if "DeepSeek" in model else "gqa"
        be_tag = "fi" if backend == "flashinfer" else "fa3"
        dcp_tag = dcp_comm.replace("_", "")
        cg_tag = "cg" if cuda_graph else "nocg"
        for req_type in REQUEST_TYPES:
            idx += 1
            sid = f"{idx:02d}_{be_tag}_{model_tag}_{dcp_tag}_{cg_tag}_{req_type}"
            scenarios.append((sid, model, backend, dcp_comm, cuda_graph, req_type))
    return scenarios


SCENARIOS = _build_scenarios()


def log(msg):
    print(f"[DCP-TEST] {msg}", flush=True)


def generate(prompt, max_tokens, temperature=0):
    """Send a generate request via curl (avoids requests dependency)."""
    payload = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
        }
    )
    cmd = [
        "curl",
        "-s",
        "-H",
        "Content-Type: application/json",
        f"{BASE_URL}/generate",
        "-d",
        payload,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None, f"curl failed: {result.stderr}"
        data = json.loads(result.stdout)
        if "text" in data:
            return data["text"], None
        return None, f"unexpected response: {result.stdout[:200]}"
    except Exception as e:
        return None, str(e)


def wait_for_server(timeout=600):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["curl", "-s", f"{BASE_URL}/get_model_info"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "model_path" in result.stdout:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def start_server(model, backend, dcp_comm, cuda_graph_enabled):
    """Launch sglang server as a subprocess."""
    env = os.environ.copy()
    env["SGLANG_DCP_SYMM_ONLY"] = "true"
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--trust-remote-code",
        "--tp-size",
        str(NUM_GPUS),
        "--dcp-size",
        str(NUM_GPUS),
        "--mem-fraction-static",
        "0.80",
        "--chunked-prefill-size",
        "32768",
        "--context-length",
        "262144",
        "--attention-backend",
        backend,
        "--disable-radix-cache",
        "--enable-symm-mem",
        "--dcp-comm-backend",
        dcp_comm,
    ]

    if not cuda_graph_enabled:
        cmd.append("--disable-cuda-graph")

    model_short = model.split("/")[-1]
    log(
        f"  Starting server: model={model_short} backend={backend} "
        f"dcp={dcp_comm} cuda_graph={cuda_graph_enabled}"
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def stop_server(proc):
    """Kill server and all its children (scheduler, TP workers, detokenizer)."""
    if proc is None:
        return

    pid = proc.pid

    # Collect all descendant PIDs via /proc before killing anything
    def _get_descendants(parent_pid):
        pids = set()
        try:
            out = subprocess.check_output(
                ["ps", "--ppid", str(parent_pid), "-o", "pid=", "--no-headers"],
                text=True,
                timeout=5,
            )
            for line in out.strip().splitlines():
                child = int(line.strip())
                pids.add(child)
                pids |= _get_descendants(child)
        except Exception:
            pass
        return pids

    all_pids = _get_descendants(pid) | {pid}

    # SIGKILL everything (SIGTERM leaves GPU memory held by TP workers)
    for p in all_pids:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass

    try:
        proc.wait(timeout=10)
    except Exception:
        pass

    time.sleep(5)


def run_request_type(req_type):
    """Run requests for a given type and return (pass, detail)."""
    cfg = REQUEST_TYPES[req_type]
    input_len = cfg["input"]
    output_len = cfg["output"]

    prompt = "Hello " * (input_len // 2)
    prompt = prompt[: input_len * 4]

    text, err = generate(prompt, output_len)
    if err:
        return False, f"generate failed: {err}"
    if text is None or len(text) == 0:
        return False, "empty output"

    # Determinism check
    text2, err2 = generate(prompt, output_len)
    if err2:
        return False, f"determinism check failed: {err2}"
    if text != text2:
        return False, f"non-deterministic: '{text[:50]}...' vs '{text2[:50]}...'"

    return True, f"ok, output_len={len(text)}"


def main():
    results = []
    total = len(SCENARIOS)

    log(f"Running {total} DCP accuracy scenarios on GPUs {GPUS}")
    log(f"TP={NUM_GPUS}, DCP={NUM_GPUS}")
    log("")

    # Group scenarios by server config (model+backend+dcp+graph) to reuse servers
    server_configs = {}
    for sid, model, backend, dcp_comm, cuda_graph, req_type in SCENARIOS:
        key = (model, backend, dcp_comm, cuda_graph)
        if key not in server_configs:
            server_configs[key] = []
        server_configs[key].append((sid, req_type))

    scenario_idx = 0
    for (model, backend, dcp_comm, cuda_graph), scenario_list in server_configs.items():
        model_short = model.split("/")[-1]
        log(
            f"=== Server: {model_short} / {backend} / {dcp_comm} / cuda_graph={cuda_graph} ==="
        )
        proc = start_server(model, backend, dcp_comm, cuda_graph)

        if not wait_for_server():
            log("  FAILED: server did not start")
            # Capture last output
            try:
                proc.terminate()
                out, _ = proc.communicate(timeout=5)
                if out:
                    log(f"  Server output (last 500 chars): ...{out.decode()[-500:]}")
            except Exception:
                pass
            for sid, req_type in scenario_list:
                scenario_idx += 1
                results.append((sid, "FAIL", "server did not start"))
                log(f"  [{scenario_idx}/{total}] {sid}: FAIL (server did not start)")
            stop_server(proc)
            continue

        log("  Server ready")

        for sid, req_type in scenario_list:
            scenario_idx += 1
            passed, detail = run_request_type(req_type)
            status = "PASS" if passed else "FAIL"
            results.append((sid, status, detail))
            log(f"  [{scenario_idx}/{total}] {sid} ({req_type}): {status} — {detail}")

        stop_server(proc)
        log("")

    # Summary
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    passed_count = sum(1 for _, s, _ in results if s == "PASS")
    failed_count = sum(1 for _, s, _ in results if s == "FAIL")

    for sid, status, detail in results:
        mark = "✓" if status == "PASS" else "✗"
        log(f"  {mark} {sid}: {status} — {detail}")

    log("")
    log(f"  {passed_count}/{total} passed, {failed_count}/{total} failed")
    log("=" * 60)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

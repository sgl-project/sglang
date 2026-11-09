"""Run on host 209 after its NPU service is released. Stops only our container."""

import json
import re
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path("/home/y00970600/personal/hisparse-lim-copysfa-gpqa-209-20261022")
NAME = "codex-lim-copysfa-gpqa-209-20261022"
info = json.loads(subprocess.check_output(["docker", "inspect", NAME], text=True))[0]
assert info["Config"]["Labels"].get("codex.task") == NAME
assert (
    json.loads((ROOT / "results/weight-transfer.json").read_text())["phase"]
    == "verified"
)
assert json.loads((ROOT / "results/checks.json").read_text())[-1]["exit_code"] == 0
smi = subprocess.check_output(["npu-smi", "info"], text=True)
pids = re.findall(r"^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|", smi, re.M)
assert not pids, "Other NPU jobs are active: " + ",".join(pids)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 8810))
status = {
    "started_unix": time.time(),
    "modes": [],
    "configuration": {
        "tp": 16,
        "ep": 16,
        "attn_dp": 2,
        "max_running_requests": 16,
        "client_threads": 8,
        "context_length": 36864,
        "max_new_tokens": 32768,
        "kv_dtype": "bfloat16",
        "mtp": False,
        "K": 2048,
        "C": 2048,
        "gpqa_examples": 198,
        "repeat": 1,
        "temperature": 0,
        "seed": 42,
    },
}


def save():
    (ROOT / "results/gpqa-ab-status.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8810" + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


save()
try:
    for mode in ("baseline", "lim_copy_sfa"):
        tag = "gpqa-209-" + mode
        row = {"mode": mode, "tag": tag, "started_unix": time.time()}
        status["modes"].append(row)
        save()
        subprocess.run(["docker", "start", NAME], stdout=subprocess.DEVNULL, check=True)
        server = subprocess.Popen(
            [
                "docker",
                "exec",
                "-e",
                "LIM_EVAL_MODE=" + mode,
                "-e",
                "LIM_EVAL_TAG=" + tag,
                NAME,
                "bash",
                "/task/launch_server.sh",
            ]
        )
        try:
            deadline = time.time() + 1800
            while time.time() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Server exited before readiness: " + tag)
                try:
                    current = request("/server_info")
                    if current.get("server_args", current).get("model_path"):
                        break
                except Exception:
                    pass
                time.sleep(10)
            else:
                raise TimeoutError("Server readiness timeout: " + tag)
            (ROOT / "results" / f"{tag}.server-info.json").write_text(
                json.dumps(current, indent=2) + "\n"
            )
            canary = request(
                "/generate",
                {
                    "text": (
                        "A short scientific note describes cells, energy, and physical laws. "
                        * 500
                    )
                    + "\nWhat is 2 + 2? Answer briefly.",
                    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
                },
            )
            (ROOT / "results" / f"{tag}.canary.json").write_text(
                json.dumps(canary, indent=2) + "\n"
            )
            assert canary.get("text"), canary
            assert canary.get("meta_info", {}).get("prompt_tokens", 0) >= 2048, (
                canary.get("meta_info")
            )
            with (ROOT / "logs" / f"{tag}.eval.log").open("w") as log:
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        NAME,
                        "python3",
                        "/task/evaluate_gpqa.py",
                        "--tag",
                        tag,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            row["phase"] = "complete"
            row["finished_unix"] = time.time()
            save()
        finally:
            subprocess.run(
                ["docker", "stop", "--time", "30", NAME],
                stdout=subprocess.DEVNULL,
                check=True,
            )
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=10)
    status["phase"] = "complete"
    status["finished_unix"] = time.time()
    save()
except BaseException as error:
    status["phase"] = "failed"
    status["error"] = repr(error)
    status["finished_unix"] = time.time()
    save()
    raise

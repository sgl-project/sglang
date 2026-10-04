"""Compare the cleanup with its main parent without changing CI assertions."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
REVISIONS = {
    "A": "1e490772e512317fab95608cbc9fb127776ae28e",
    "B": "a42c97419b70512264442893ec03b46477af45ef",
}
TASK = Path(tempfile.mkdtemp(prefix="ltx23-cleanup-ab-"))
CHECKOUTS = {}

CHILD = r'''
import json
import pathlib
import sys
import pytest
import torch
import sglang
from sglang.multimodal_gen.test.server.test_server_common import DiffusionServerBase

expected_source = pathlib.Path(sys.argv[2]).resolve()
assert pathlib.Path(sglang.__file__).resolve().is_relative_to(expected_source)
original = DiffusionServerBase._test_diffusion_request

def cold_request(self, *args, **kwargs):
    devices = []
    for device in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device)
        eviction = torch.empty(props.L2_cache_size * 5, device=device, dtype=torch.uint8)
        eviction.fill_(1)
        torch.cuda.synchronize(device)
        devices.append(dict(device=device, l2_bytes=props.L2_cache_size,
                            eviction_bytes=eviction.numel()))
        del eviction
    print("COLD_REQUEST", json.dumps(dict(devices=devices, source=sglang.__file__)), flush=True)
    return original(self, *args, **kwargs)

DiffusionServerBase._test_diffusion_request = cold_request
sys.exit(pytest.main([sys.argv[1], "-k", "ltx_2_3_two_stage_ti2v_2gpus", "-x", "-s"]))
'''


def gpu_state():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version,power.limit,clocks.sm,temperature.gpu", "--format=csv,noheader"],
        text=True,
    )


def main():
    print("AB_ENVIRONMENT", json.dumps({
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "triton", "flashinfer-python")},
        "gpu": gpu_state(),
        "cold_l2_scope": "5x L2 eviction on both GPUs before each request, outside request timing; not before each internal kernel",
        "revisions": REVISIONS,
    }), flush=True)
    results = []
    try:
        for variant, revision in REVISIONS.items():
            subprocess.run(["git", "fetch", "--depth=1", "origin", revision], cwd=ROOT, check=True)
            checkout = TASK / variant
            subprocess.run(["git", "worktree", "add", "--detach", str(checkout), revision], cwd=ROOT, check=True)
            CHECKOUTS[variant] = checkout
        for index, variant in enumerate("ABBAABBA"):
            checkout = CHECKOUTS[variant]
            out = TASK / str(index)
            out.mkdir()
            env = dict(os.environ, PYTHONPATH=str(checkout / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
            test = checkout / "python/sglang/multimodal_gen/test/server/test_server_2_gpu.py"
            before = gpu_state()
            proc = subprocess.run(
                [sys.executable, "-c", CHILD, str(test), str(checkout)],
                cwd=out, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=900,
            )
            (out / "test.log").write_text(proc.stdout)
            metrics = out / "diffusion-results.json"
            entry = {
                "index": index, "variant": variant, "revision": REVISIONS[variant],
                "exit_code": proc.returncode, "gpu_before": before,
                "metrics": json.loads(metrics.read_text()) if metrics.exists() else [],
                "cold_request": [line for line in proc.stdout.splitlines() if "COLD_REQUEST" in line],
                "consistency": [line for line in proc.stdout.splitlines() if "[Consistency Check]" in line],
            }
            results.append(entry)
            print("AB_RESULT", json.dumps(entry), flush=True)
            if proc.returncode:
                print(proc.stdout[-8000:], flush=True)
            (TASK / "results.json").write_text(json.dumps(results, indent=2))
            if not entry["metrics"] or not entry["cold_request"]:
                raise RuntimeError("Incomplete measurement; inspect the recorded test failure")
        print("AB_REPORT", json.dumps(results), flush=True)
        return int(any(entry["exit_code"] for entry in results))
    finally:
        for checkout in CHECKOUTS.values():
            subprocess.run(["git", "worktree", "remove", str(checkout)], cwd=ROOT, check=False)


if __name__ == "__main__":
    sys.exit(main())

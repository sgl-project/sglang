"""Diagnostic candidate only; all original test outcomes are retained."""
import json
import subprocess
import sys
from pathlib import Path
print("CANDIDATE_CHECKOUT", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), flush=True)
commands = [
    [sys.executable, "-m", "pytest", "sglang/multimodal_gen/test/server/test_server_2_gpu.py", "-v", "-s", "--tb=short", "-k", "ltx_2_5_diffusion_decoder_2gpus"],
    [sys.executable, "/sglang-checkout/test/manual/amd_flex_cold_l2_probe.py"],
]
results = []
for command in commands:
    print("CANDIDATE_BEGIN", json.dumps(command), flush=True)
    code = subprocess.run(command).returncode
    results.append(code)
    print("CANDIDATE_END", code, flush=True)
print("CANDIDATE_RESULTS", results, flush=True)
sys.exit(1 if any(results) else 0)

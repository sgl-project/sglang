"""One-shot diagnosis; original pytest outcomes are preserved."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

print("DIAG_CHECKOUT", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), flush=True)
for tool in ("ffmpeg", "ffprobe"):
    assert shutil.which(tool), tool
    subprocess.run([tool, "-version"], check=True)

commands = [
    [sys.executable, "-m", "pytest", "sglang/multimodal_gen/test/unit/test_diffusion_import_isolation.py", "-v", "-s"],
    [sys.executable, "-m", "pytest", "sglang/multimodal_gen/test/server/test_server_2_gpu.py", "-v", "-s", "--tb=short", "-k", "minimax_h3_t2va or minimax_h3_ref2va_video_audio"],
    [sys.executable, "-m", "pytest", "sglang/multimodal_gen/test/server/test_server_2_gpu.py", "-v", "-s", "--tb=short", "-k", "ltx_2_5_diffusion_decoder_2gpus"],
]
results = []
for command in commands:
    print("DIAG_BEGIN", json.dumps(command), flush=True)
    code = subprocess.run(command).returncode
    results.append(code)
    print("DIAG_END", code, flush=True)
    for path in sorted(Path("/tmp").glob("ltx-stacks-*.log")):
        print("DIAG_STACK", path, flush=True)
        print(path.read_text(), flush=True)
print("DIAG_RESULTS", json.dumps(results), flush=True)
sys.exit(1 if any(results) else 0)

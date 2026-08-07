import os
import subprocess
import sys
from pathlib import Path


def test_lightweight_runtime_import_does_not_load_torch():
    python_root = Path(__file__).resolve().parents[5]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_root)
    env["SGLANG_LIGHTWEIGHT_RUNTIME"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr

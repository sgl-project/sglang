"""Server start on ROCm must not need aiter.

Importing the HTTP entrypoint reaches the Quark schemes that bound aiter at
module level behind ``is_hip()`` alone; an aiter-less ROCm install (the usual
setup on Radeon/RDNA) failed there at import time.
"""

import subprocess
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="base-b", runner_config="1-gpu-small-amd")


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="is_hip() gates the imports under test",
)
def test_http_server_imports_without_aiter() -> None:
    # None in sys.modules makes `import aiter` raise and find_spec return None.
    code = (
        "import sys; sys.modules['aiter'] = None; "
        "import sglang.srt.entrypoints.http_server"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

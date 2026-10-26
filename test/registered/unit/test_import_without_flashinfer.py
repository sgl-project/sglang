"""Server start must not need FlashInfer.

Importing the HTTP entrypoint reaches every module that bound FlashInfer at
module level behind ``is_cuda()`` alone (sampler, dflash_utils, fp8_utils,
attention/vision); a FlashInfer-less CUDA image failed there at import time.
"""

import subprocess
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="is_cuda() gates the imports under test"
)
def test_http_server_imports_without_flashinfer() -> None:
    # None in sys.modules makes `import flashinfer` raise and find_spec return None.
    code = (
        "import sys; sys.modules['flashinfer'] = None; "
        "import sglang.srt.entrypoints.http_server"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-3000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")


class TestKDACPUImport(CustomTestCase):
    def test_cpu_import_executes_extend_and_decode(self):
        """A fresh CPU process must bind chunk KDA without a patched import."""
        code = textwrap.dedent(
            """
            import torch
            from sgl_kernel.mamba import chunk_kda_cpu
            from sglang.srt.layers.attention.linear.kernels import kda_triton

            torch.set_num_threads(2)
            assert kda_triton.is_cpu()
            assert kda_triton.chunk_kda is chunk_kda_cpu
            kernel = kda_triton.TritonKDAKernel()
            q = torch.zeros(1, 2, 1, 64, dtype=torch.bfloat16)
            beta = torch.zeros(1, 2, 1, dtype=torch.bfloat16)
            offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
            indices = torch.tensor([1, 0], dtype=torch.int32)
            params = dict(
                q=q, k=q, v=q,
                A_log=torch.zeros(1), dt_bias=torch.zeros(64),
                cache_indices=indices, query_start_loc=offsets,
            )
            for decode in (False, True):
                state = torch.ones(2, 1, 64, 64)
                if decode:
                    out = kernel.decode(
                        **params, a=q.flatten(2), b=beta, ssm_states=state
                    )
                else:
                    out = kernel.extend(
                        **params, g=q, beta=beta, beta_is_raw=True, ssm_states=state
                    )
                torch.testing.assert_close(out, q, rtol=0, atol=0)
                torch.testing.assert_close(state, torch.full_like(state, 0.5))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "SGLANG_USE_CPU_ENGINE": "1"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

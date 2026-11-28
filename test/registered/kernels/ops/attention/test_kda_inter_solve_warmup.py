import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-large")


def _check_cold_cache(cache_dir):
    from sglang.kernels.ops.attention.fla.chunk_intra import precompile_k3_inter_solve
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    def binaries():
        return set(
            Path(cache_dir).rglob("chunk_kda_fwd_kernel_inter_solve_fused.cubin")
        )

    heads, dim = 12, 128
    precompile_k3_inter_solve(num_heads=heads, head_dim=dim, safe_gate=True)
    warmed = binaries()
    assert warmed, "startup did not compile inter_solve"

    # 21 chunks x 12 heads = 252 (fused), 22 x 12 = 264 (unfused).
    for tokens in (1344, 1345):
        shape = (1, tokens, heads, dim)
        q = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        chunk_kda(
            q=q,
            k=q,
            v=torch.zeros_like(q),
            g=q,
            beta=q.new_zeros((1, tokens, heads)),
            initial_state=q.new_zeros((1, heads, dim, dim), dtype=torch.float32),
            initial_state_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
            cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
            A_log=q.new_zeros((heads,), dtype=torch.float32),
            dt_bias=q.new_zeros((heads * dim,), dtype=torch.float32),
            lower_bound=-5.0,
            use_qk_l2norm_in_kernel=True,
            beta_is_raw=True,
        )
        torch.cuda.synchronize()
        assert binaries() == warmed, f"unexpected inter_solve JIT at {tokens} tokens"


class TestKDAInterSolveWarmup(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
        "Kimi K3 inter_solve startup precompile targets Hopper",
    )
    def test_first_prefill_reuses_both_specializations(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            worker = multiprocessing.get_context("spawn").Process(
                target=_check_cold_cache, args=(cache_dir,)
            )
            # Set before spawning so even import-time caches see the empty directory.
            with patch.dict(os.environ, {"TRITON_CACHE_DIR": cache_dir}):
                worker.start()
            worker.join()
            self.assertEqual(worker.exitcode, 0)


if __name__ == "__main__":
    unittest.main()

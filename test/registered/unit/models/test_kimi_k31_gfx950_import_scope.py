"""Kimi-K3.1's gfx950 modules load only on gfx950."""

import json
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# The shared modules Kimi-K3.1's gfx950 support calls into from its hook points.
SHARED_MODULES = [
    "sglang.srt.arg_groups.model_hook",
    "sglang.srt.arg_groups.speculative_hook",
    "sglang.srt.distributed.bootstrap",
    "sglang.srt.layers.attention.dsa.dsa_dcp_sharded",
    "sglang.srt.layers.attention.dsa.dsa_indexer",
    "sglang.srt.layers.attention.dsa_backend",
    "sglang.srt.layers.attention.linear.kda_backend",
    "sglang.srt.layers.attention.triton_backend",
    "sglang.srt.layers.attn_residual",
    "sglang.srt.layers.dcp.comm",
    "sglang.srt.layers.layer_boundary.residual.attn_bank",
    "sglang.srt.layers.quantization.mxfp4",
    "sglang.srt.mem_cache.hicache_auto_size",
    "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler",
    "sglang.srt.mem_cache.memory_pool",
    "sglang.srt.mem_cache.pool_host.mamba",
    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
    "sglang.srt.models.dflash",
    "sglang.srt.models.kimi_k3",
    "sglang.srt.speculative.dflash_worker_v2",
]

# Its gfx950-only modules. Not listed: unquant_hip, which forward_mla_rocm imports
# on every platform, and kimi_k3_amd.moe, which is KimiK3MoE's base class.
GFX950_MODULES = [
    "sglang.srt.distributed.bootstrap_rocm",
    "sglang.srt.layers.attention.dsa.dsa_backend_hip",
    "sglang.srt.layers.attention.dsa.dsa_dcp_sharded_hip",
    "sglang.srt.layers.attention.dsa.dsa_indexer_hip",
    "sglang.srt.layers.attention.dsa.indexer_prologue_hip",
    "sglang.srt.layers.attention.linear.kda_backend_hip",
    "sglang.srt.layers.attention.triton_draft_hip",
    "sglang.srt.layers.attn_residual_hip",
    "sglang.srt.layers.layer_boundary.residual.attn_bank_hip",
    "sglang.srt.mem_cache.index_key_cache_hip",
    "sglang.srt.mem_cache.pool_host.rocm_budget",
    "sglang.srt.models.kimi_k3_amd.env_defaults",
    "sglang.srt.models.kimi_k3_amd.kimi_k3_hip",
]

# A fresh interpreter, so modules other tests imported cannot hide a leak.
_PROBE = """
import json, sys
import torch
# a non-ROCm build, whatever torch this host has
torch.version.hip = None
torch.cuda.is_available = lambda: False
for name in {shared!r}:
    __import__(name)
print(json.dumps(sorted(m for m in {gfx950!r} if m in sys.modules)))
"""


class TestKimiK31Gfx950ImportScope(CustomTestCase):
    def test_shared_modules_load_no_gfx950_module(self):
        """A module-top import of a gfx950 module in a shared file loads it on CUDA too."""
        probe = _PROBE.format(shared=SHARED_MODULES, gfx950=GFX950_MODULES)
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        loaded = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(loaded, [], f"loaded off gfx950: {loaded}")


if __name__ == "__main__":
    unittest.main()

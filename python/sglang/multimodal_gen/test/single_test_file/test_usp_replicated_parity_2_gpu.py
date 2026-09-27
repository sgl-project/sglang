"""USPAttention replicated prefix/suffix must match a single-rank reference.

The suffix path keeps the replicated tokens at the sequence tail so every
query scans K/V in the same order as a single rank — bitwise-stable across SP
degrees. A rotate-to-front implementation is numerically valid but reorders
the reduction, and few-step (turbo) models amplify that into visible drift;
this test pins the order-preserving behavior.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_usp_replicated_parity_2_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 2


def _worker() -> int:
    from types import SimpleNamespace

    import torch.nn.functional as F

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )

    import sglang.multimodal_gen.runtime.layers.attention.layer as L

    L.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)

    class Sdpa:
        def __init__(self, scale):
            self.scale = scale

        def forward(self, q, k, v, _ctx):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scale,
            ).transpose(1, 2)

    B, SHARD, H, D, REP = 1, 64, 30, 128, 6
    S = SHARD * world
    scale = D**-0.5
    torch.manual_seed(0)
    dev = torch.device(f"cuda:{rank}")
    qf = torch.randn(B, S + REP, H, D, device=dev, dtype=torch.bfloat16)
    kf = torch.randn(B, S + REP, H, D, device=dev, dtype=torch.bfloat16)
    vf = torch.randn(B, S + REP, H, D, device=dev, dtype=torch.bfloat16)

    ref = F.scaled_dot_product_attention(
        qf.transpose(1, 2), kf.transpose(1, 2), vf.transpose(1, 2), scale=scale
    ).transpose(1, 2)

    attn = L.USPAttention.__new__(L.USPAttention)
    attn.causal = False
    attn.softmax_scale = scale
    attn.attn_impl = Sdpa(scale)
    attn.skip_sequence_parallel = False
    attn.enable_packed_qkv_input_a2a = False
    attn.allow_cudnn_sdp = False
    attn.backend = L.AttentionBackendEnum.TORCH_SDPA
    attn.dtype = torch.bfloat16
    attn.dropout_p = 0.0
    attn.sp_attention_mode = "ulysses"
    attn.sp_attention_mode_is_auto = False

    failures = []
    sl = slice(rank * SHARD, (rank + 1) * SHARD)

    out = attn.forward(
        torch.cat([qf[:, sl], qf[:, S:]], dim=1),
        torch.cat([kf[:, sl], kf[:, S:]], dim=1),
        torch.cat([vf[:, sl], vf[:, S:]], dim=1),
        num_replicated_suffix=REP,
    )
    exp = torch.cat([ref[:, sl], ref[:, S:]], dim=1)
    if not torch.equal(out, exp):
        d = (out.float() - exp.float()).abs()
        failures.append(f"suffix not bitwise: mae={d.mean():.3e} max={d.max():.3e}")

    out_p = attn.forward(
        torch.cat([qf[:, S:], qf[:, sl]], dim=1),
        torch.cat([kf[:, S:], kf[:, sl]], dim=1),
        torch.cat([vf[:, S:], vf[:, sl]], dim=1),
        num_replicated_prefix=REP,
    )
    exp_p = torch.cat([ref[:, S:], ref[:, sl]], dim=1)
    dp = (out_p.float() - exp_p.float()).abs()
    if dp.max().item() > 1e-2:
        failures.append(f"prefix drift: mae={dp.mean():.3e} max={dp.max():.3e}")

    for f in failures:
        print(f"FAILURE rank{rank}: {f}", flush=True)
    return 1 if failures else 0


class TestUSPReplicatedParity(CustomTestCase):
    def test_replicated_parity_two_ranks(self):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        procs = []
        for rank in range(_WORLD):
            env = os.environ.copy()
            env.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(_WORLD),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29751",
                }
            )
            procs.append(
                subprocess.Popen(
                    [sys.executable, __file__],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = [p.communicate(timeout=300)[0] for p in procs]
        codes = [p.returncode for p in procs]
        if any(codes):
            self.fail("worker failed:\n" + "\n".join(outputs))


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    unittest.main()

"""Ulysses torch SDPA must match a single rank bit for bit on short sequences.

After the all-to-all each rank holds half the heads over the full sequence.
PyTorch's flash forward may then pick split-KV where the unsharded layer does
not; for "exact" requests the SDPA backend pads query rows to keep the
unsharded kernel.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_usp_sdpa_sharded_heads_2_gpu.py
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
_ALL_SKIPPED = 77
# (heads, sequence): the unsharded layer stays unsplit, half the heads split.
_CASES = ((24, 1024), (12, 1536))


def _worker() -> int:
    from types import SimpleNamespace

    import torch.nn.functional as F

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )

    import sglang.multimodal_gen.runtime.layers.attention.layer as L
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    set_global_server_args(
        SimpleNamespace(
            attention_backend="torch_sdpa",
            attention_backend_config=None,
            kv_gather_degree=1,
            sp_split_auto=False,
        )
    )
    torch.set_default_dtype(torch.bfloat16)

    request = set_forward_context(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=SimpleNamespace(quality="exact"),
    )
    request.__enter__()

    D = 128
    scale = D**-0.5
    dev = torch.device(f"cuda:{rank}")
    failures = []
    checked = 0
    for heads, seq in _CASES:
        torch.manual_seed(heads + seq)
        qf, kf, vf = (
            torch.randn(1, seq, heads, D, device=dev, dtype=torch.bfloat16)
            for _ in range(3)
        )
        ref = F.scaled_dot_product_attention(
            qf.transpose(1, 2), kf.transpose(1, 2), vf.transpose(1, 2), scale=scale
        ).transpose(1, 2)

        # global_num_heads defaults to num_heads, the Ulysses-only case.
        attn = L.USPAttention(
            num_heads=heads,
            head_size=D,
            causal=False,
            required_attention_backend=AttentionBackendEnum.TORCH_SDPA,
        )
        local = qf[:, :, : heads // world].transpose(1, 2)
        if attn.attn_impl._unsplit_query_len(local, seq) == seq:
            print(f"SKIP: heads={heads} seq={seq} does not split here", flush=True)
            continue

        checked += 1
        shard = seq // world
        sl = slice(rank * shard, (rank + 1) * shard)
        out = attn.forward(qf[:, sl], kf[:, sl], vf[:, sl])
        if not torch.equal(out, ref[:, sl]):
            d = (out.float() - ref[:, sl].float()).abs()
            failures.append(
                f"heads={heads} seq={seq} not bitwise: "
                f"mismatched={(d > 0).sum().item()} max={d.max():.3e}"
            )

    for f in failures:
        print(f"FAILURE rank{rank}: {f}", flush=True)
    if failures:
        return 1
    return 0 if checked else _ALL_SKIPPED


class TestUSPSDPAShardedHeads(CustomTestCase):
    def test_ulysses_sdpa_matches_single_rank(self):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        from sglang.srt.utils.network import get_free_port_below_ephemeral

        port = str(get_free_port_below_ephemeral())
        procs = []
        for rank in range(_WORLD):
            env = os.environ.copy()
            env.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(_WORLD),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": port,
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
        if all(code == _ALL_SKIPPED for code in codes):
            self.skipTest("this GPU has enough SMs that no case splits")
        if any(codes):
            self.fail("worker failed:\n" + "\n".join(outputs))


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    unittest.main()

"""A folded encoder must run SRT collectives on its bound TP group."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD_SIZE = 2


def _worker() -> int:
    from sglang.multimodal_gen.runtime.distributed import (
        cleanup_dist_env_and_memory,
        get_tp_group,
        get_world_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.models.encoders.base import (
        EncoderTensorParallelMixin,
    )
    from sglang.srt.distributed import parallel_state as srt_parallel_state
    from sglang.srt.layers.linear import RowParallelLinear

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
    )
    initialize_model_parallel(
        tensor_parallel_degree=1,
        sequence_parallel_degree=world_size,
        ulysses_degree=world_size,
        ring_degree=1,
    )

    class FoldedEncoder(EncoderTensorParallelMixin, nn.Module):
        def __init__(self):
            super().__init__()
            self.bind_encoder_tp_group(get_world_group())
            self.proj = RowParallelLinear(
                input_size=8,
                output_size=6,
                bias=False,
                tp_rank=rank,
                tp_size=world_size,
                params_dtype=torch.float32,
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            local_inputs = inputs.chunk(world_size, dim=-1)[rank].contiguous()
            output, _ = self.proj(local_inputs)
            return output

    full_weight = torch.arange(48, dtype=torch.float32, device=device).reshape(6, 8)
    full_weight = (full_weight - 23.5) / 32
    inputs = torch.arange(24, dtype=torch.float32, device=device).reshape(3, 8) / 8
    model = FoldedEncoder().to(device).eval()
    with torch.no_grad():
        model.proj.weight.copy_(full_weight[:, rank * 4 : (rank + 1) * 4].contiguous())
        expected = F.linear(inputs, full_weight)
        actual = model(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert get_tp_group().world_size == 1
    assert srt_parallel_state.get_tp_group().world_size == 1
    assert srt_parallel_state.get_attn_tp_group().world_size == 1

    if rank == 0:
        print("ENCODER_FOLD_SRT_LINEAR_PARITY PASS", flush=True)
    torch.distributed.barrier()
    cleanup_dist_env_and_memory()
    return 0


class TestEncoderFoldSrtLinearTwoGpu(CustomTestCase):
    def test_folded_srt_linear_matches_unsharded_reference(self):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest(f"needs {_WORLD_SIZE} GPUs")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD_SIZE}",
                "--master-port=29618",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "folded SRT linear output diverged")
        self.assertIn("ENCODER_FOLD_SRT_LINEAR_PARITY PASS", proc.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()

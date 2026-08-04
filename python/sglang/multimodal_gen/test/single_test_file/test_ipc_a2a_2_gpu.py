"""The CUDA-IPC all-to-all must match NCCL bit for bit.

This transport only activates on 2 ranks with peer-to-peer access, so nothing in
the single-GPU suite reaches it. Without this test a regression in the IPC path
lands silently.

Runs the whole USPAttention A2A family twice per shape -- once over NCCL, once
over IPC -- and requires identical bytes:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_ipc_a2a_2_gpu.py
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
    """One rank: compare every A2A path against its NCCL result."""
    import torch.distributed as dist

    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=_WORLD, ulysses_degree=_WORLD
    )

    from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
        IPC_A2A,
    )
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all,
        _usp_output_all_to_all,
    )

    if not torch.cuda.can_device_access_peer(rank, 1 - rank):
        print("SKIP no peer-to-peer access between the two devices", flush=True)
        return 0

    # b, s_local, h_global, d -- h_global must split across the two ranks
    shapes = [
        (1, 256, 8, 64),
        (1, 1152, 24, 128),  # qwen-image 1024x1024 class
        (2, 64, 4, 64),  # batched, and a short sequence like a warmup prompt
    ]
    failures = []
    for b, s_local, h_global, d in shapes:
        torch.manual_seed(1234)  # same input on both ranks' RNG, sliced per rank
        full = torch.randn(
            b, s_local * _WORLD, h_global, d, dtype=torch.bfloat16, device="cuda"
        )
        x_in = full.narrow(1, rank * s_local, s_local).contiguous()

        def both_paths(fn, arg, **kw):
            envs.SGLANG_DIFFUSION_IPC_A2A = False
            nccl = fn(arg, **kw)
            envs.SGLANG_DIFFUSION_IPC_A2A = True
            ipc = fn(arg, **kw)
            envs.SGLANG_DIFFUSION_IPC_A2A = False
            return nccl, ipc

        nccl_out, ipc_out = both_paths(_usp_input_all_to_all, x_in, head_dim=2)
        if not torch.equal(nccl_out, ipc_out):
            failures.append(f"input a2a {(b, s_local, h_global, d)}")

        # the output A2A consumes [b, s_global, h_local, d]
        x_out = torch.randn(
            b,
            s_local * _WORLD,
            h_global // _WORLD,
            d,
            dtype=torch.bfloat16,
            device="cuda",
        )
        nccl_out, ipc_out = both_paths(_usp_output_all_to_all, x_out, head_dim=2)
        if not torch.equal(nccl_out, ipc_out):
            failures.append(f"output a2a {(b, s_local, h_global, d)}")

    # AllToAll4D is a second entry point (stacked-qkv UlyssesAttention: zimage's
    # secondary attention, wan-VSA, hunyuanvideo) and routes through the same
    # exchange, so it needs its own parity check.
    from sglang.multimodal_gen.runtime.distributed.communication_op import (
        sequence_model_parallel_all_to_all_4D as all_to_all_4D,
    )

    # count exchanges, not staging keys: the key is (n_local, n_peer, dtype),
    # so these shapes reuse the buffers the earlier arms already allocated
    calls_before_a2a4d = IPC_A2A.calls
    for b, s_local, h_global, d in shapes:
        torch.manual_seed(4321)
        for scatter_dim in (1, 2):
            if scatter_dim == 2:
                x = torch.randn(
                    b, s_local, h_global, d, dtype=torch.bfloat16, device="cuda"
                )
            else:
                x = torch.randn(
                    b,
                    s_local * _WORLD,
                    h_global // _WORLD,
                    d,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            envs.SGLANG_DIFFUSION_IPC_A2A = False
            nccl = all_to_all_4D(x, scatter_dim=scatter_dim, gather_dim=3 - scatter_dim)
            envs.SGLANG_DIFFUSION_IPC_A2A = True
            ipc = all_to_all_4D(x, scatter_dim=scatter_dim, gather_dim=3 - scatter_dim)
            envs.SGLANG_DIFFUSION_IPC_A2A = False
            if nccl.shape != ipc.shape:
                failures.append(
                    f"a2a4d scatter={scatter_dim} shape {tuple(nccl.shape)} != "
                    f"{tuple(ipc.shape)} for {(b, s_local, h_global, d)}"
                )
            elif not torch.equal(nccl, ipc):
                failures.append(
                    f"a2a4d scatter={scatter_dim} {(b, s_local, h_global, d)}"
                )

    # A parity check that never reached the IPC branch would pass while proving
    # nothing, so require that exchanges actually happened.
    if IPC_A2A.calls == calls_before_a2a4d:
        failures.append(
            f"AllToAll4D never took the IPC path (exchange count stayed at "
            f"{calls_before_a2a4d})"
        )

    # the transport must survive a shape it has never staged before, and the
    # eviction that a capped cache performs
    envs.SGLANG_DIFFUSION_IPC_A2A = True
    IPC_A2A.max_buffers = 1
    for s_local in (32, 48, 64):
        x = torch.randn(1, s_local, 8, 64, dtype=torch.bfloat16, device="cuda")
        if _usp_input_all_to_all(x, head_dim=2) is None:
            failures.append(f"eviction path returned None at s_local={s_local}")
    envs.SGLANG_DIFFUSION_IPC_A2A = False

    # A comparison where both arms quietly fell back to NCCL would pass while
    # testing nothing, so require evidence the transport actually ran.
    if not IPC_A2A.inited or IPC_A2A.failed or not IPC_A2A.staging:
        failures.append(
            f"IPC never engaged (inited={IPC_A2A.inited} failed={IPC_A2A.failed} "
            f"staged={len(IPC_A2A.staging)})"
        )

    # A timeout must retire the transport on BOTH ranks. When only the rank that
    # timed out switched to NCCL, it posted an all_to_all its peer -- still on
    # IPC -- never posted, and the NCCL watchdog took the process down (seen on
    # wan2_2_t2v_a14b_2gpu in CI). Rank 0 waits for a signal rank 1 never sends,
    # so rank 1 may only learn of it through the peer-side flag write.
    IPC_A2A.budget_ns = 5 * 1000 * 1000
    if rank == 0:
        IPC_A2A.signal_and_wait()
    torch.cuda.synchronize()
    dist.barrier()
    if IPC_A2A.timed_out.item() == 0:
        failures.append(f"rank{rank} never saw the timeout flag")
    try:
        IPC_A2A.check_timeout()
        failures.append(f"rank{rank} check_timeout did not raise after a timeout")
    except RuntimeError:
        pass

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures}", flush=True)
    if rank == 0:
        print(
            f"IPC_A2A_PARITY {'FAIL' if verdict.item() else 'PASS'} "
            f"(staged keys={len(IPC_A2A.staging)})",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestIpcA2ATwoGpu(CustomTestCase):
    def test_ipc_matches_nccl_bitwise(self):
        # CUDA only: the transport opens torch IPC handles through libcudart and
        # cudaDeviceEnablePeerAccess. On ROCm/NPU it correctly refuses and both
        # arms run over NCCL, which the evidence assertion below reads -- rightly
        # -- as "the transport never engaged". That is unsupported, not broken.
        if not current_platform.is_cuda():
            self.skipTest("CUDA-IPC transport is unavailable on this platform")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD}",
                "--master-port=29517",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "IPC all-to-all diverged from NCCL")
        self.assertIn("IPC_A2A_PARITY PASS", proc.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()

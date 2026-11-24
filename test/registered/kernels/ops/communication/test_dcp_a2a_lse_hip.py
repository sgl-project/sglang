"""ROCm gfx950 fused XGMI push DCP all-to-all + LSE combine parity & graph replay tests."""

import multiprocessing
import socket
import time
import unittest

import torch
import torch.distributed as dist
from sglang.kernels.ops.attention.dcp_kernels import (
    dcp_lse_combine_triton,
    dcp_pack_a2a_send,
)
from sglang.kernels.ops.communication.dcp_a2a_lse_hip import (
    can_run_dcp_a2a_lse_hip,
    create_dcp_ll_comm,
    dcp_a2a_lse_hip,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")

WORLD_SIZE = 8
H_LOCAL = 12


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _ref_a2a_lse(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    is_lse_base_on_e: bool,
) -> torch.Tensor:
    n = dist.get_world_size()
    b, h, d = cp_attn_out.shape
    h_per = h // n
    send = torch.empty(
        (n, b, h_per, d + 2), dtype=cp_attn_out.dtype, device=cp_attn_out.device
    )
    recv = torch.empty_like(send)
    dcp_pack_a2a_send(
        cp_attn_out,
        cp_attn_lse,
        send[:, :, :, :d],
        send.view(torch.float32)[:, :, :, d // 2],
    )
    dist.all_to_all_single(
        recv.reshape(-1).view(torch.uint8),
        send.reshape(-1).view(torch.uint8),
    )
    out, _ = dcp_lse_combine_triton(
        recv[:, :b, :, :d],
        recv.view(torch.float32)[:, :b, :, d // 2],
        is_lse_base_on_e=is_lse_base_on_e,
    )
    return out


def _worker(rank: int, world_size: int, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        cpu_group = dist.new_group(backend="gloo")
        h = world_size * H_LOCAL
        for head_dim in (128, 512):
            comm = create_dcp_ll_comm(
                cpu_group,
                device,
                world_size=world_size,
                head_dim=head_dim,
                num_local_heads=H_LOCAL,
                max_rows=192,
            )
            assert comm is not None
            bad_rows = torch.empty(
                (193, h, head_dim), dtype=torch.bfloat16, device=device
            )
            bad_lse = torch.empty((193, h), dtype=torch.float32, device=device)
            assert not can_run_dcp_a2a_lse_hip(comm, bad_rows, bad_lse)
            assert not can_run_dcp_a2a_lse_hip(None, bad_rows[:1], bad_lse[:1])

            for dtype in (torch.bfloat16, torch.float16):
                tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
                for is_base_e in (True, False):
                    # Includes B=128 (1536 slots > max_grid on 256-CU gfx950) to
                    # exercise multi-iteration persistent slot scheduling.
                    for b in (1, 6, 48, 128):
                        torch.manual_seed(1000 + rank * 37 + b)
                        x = torch.randn((b, h, head_dim), dtype=dtype, device=device)
                        lse = (
                            torch.randn((b, h), dtype=torch.float32, device=device)
                            * 3.0
                        )
                        if b >= 6:
                            lse[0, :] = -float("inf")
                            lse[1, :H_LOCAL] = float("nan")
                            lse[2, :H_LOCAL] = float("inf")
                            if rank % 2 == 0:
                                lse[3, :] = -float("inf")
                        ref = _ref_a2a_lse(x, lse, is_base_e)
                        got = dcp_a2a_lse_hip(
                            x, lse, comm=comm, is_lse_base_on_e=is_base_e
                        )
                        assert not torch.isnan(got).any()
                        err = (got.float() - ref.float()).abs().max().item()
                        assert err <= tol, (
                            f"d={head_dim} dtype={dtype} base_e={is_base_e} b={b}: "
                            f"max_err={err} > {tol}"
                        )

            x_buf = torch.randn((24, h, head_dim), dtype=torch.bfloat16, device=device)
            lse_buf = torch.randn((24, h), dtype=torch.float32, device=device)
            _ = dcp_a2a_lse_hip(x_buf, lse_buf, comm=comm, is_lse_base_on_e=False)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_g = dcp_a2a_lse_hip(
                    x_buf, lse_buf, comm=comm, is_lse_base_on_e=False
                )
            for step in range(6):
                torch.manual_seed(5000 + step * 100 + rank)
                x_buf.copy_(torch.randn_like(x_buf))
                lse_buf.copy_(torch.randn_like(lse_buf) * 2.0)
                graph.replay()
                ref_g = _ref_a2a_lse(x_buf, lse_buf, False)
                assert (out_g.float() - ref_g.float()).abs().max().item() <= 2e-2
    finally:
        dist.destroy_process_group()


class TestDcpA2ALseHip(CustomTestCase):
    @unittest.skipUnless(
        is_hip() and is_gfx95_supported() and torch.cuda.device_count() >= WORLD_SIZE,
        "Requires 8 gfx950 GPUs",
    )
    def test_dcp_a2a_lse_hip_parity_and_graph_replay(self) -> None:
        port = _get_open_port()
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(rank, WORLD_SIZE, port))
            for rank in range(WORLD_SIZE)
        ]
        for proc in procs:
            proc.start()
        deadline = time.monotonic() + 420.0
        timed_out = False
        for proc in procs:
            remaining = max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                timed_out = True
                break
        if timed_out:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            for proc in procs:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.kill()
            self.fail("test_dcp_a2a_lse_hip timed out after 420s")
        for proc in procs:
            self.assertEqual(proc.exitcode, 0)


if __name__ == "__main__":
    unittest.main()

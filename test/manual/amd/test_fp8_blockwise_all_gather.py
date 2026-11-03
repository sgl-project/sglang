"""Eight-GPU FP8 blockwise AG/CK QKV smoke, without loading a model.

SGLANG_USE_AITER=1 SGLANG_USE_AITER_CK_BLOCKSCALE_BPRESHUFFLE=1 \
  python test/manual/amd/test_fp8_blockwise_all_gather.py

Checks two independent TP4 groups, one idle DP, residual/no-residual quant,
RCCL/custom AG, and graph replay with changed inputs. Needs matching AITER.
"""

import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=8
    )
    groups = [dist.new_group(list(range(4))), dist.new_group(list(range(4, 8)))]
    group = groups[rank // 4]
    import aiter
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_group_quant
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.quantization.fp8_blockwise_all_gather import (
        Fp8BlockwisePacket,
        all_gather_fp8_blockwise,
    )
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear

    nccl = PyNcclCommunicator(group, rank)
    ca = CustomAllreduce(group, rank, max_size=256 * 1024 * 1024)
    assert not ca.disabled
    coordinator = GroupCoordinator.__new__(GroupCoordinator)
    coordinator.world_size = 4
    coordinator.pynccl_comm = nccl
    coordinator.is_symmetric_memory_enabled = lambda: False
    coordinator.debug_check_symmetric_mempool = lambda *args: None
    torch.manual_seed(12345 + rank)
    h, g, n = 4096, 32, 3712
    weight = shuffle_weight(
        torch.randn((n, h), device="cuda", dtype=torch.bfloat16).to(aiter.dtypes.fp8),
        (16, 16),
    )
    weight_scale = torch.full((n // 128, g), 0.01, device="cuda", dtype=torch.float32)
    for custom in (False, True):
        coordinator.ca_comm = ca if custom else None
        for row_counts in ((0, 17), (17, 0), (17, 17), (4096, 4096)):
            m = row_counts[rank // 4]
            x = torch.randn((m, h), device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x) if m != 17 else None
            norm_weight = torch.randn(h, device="cuda", dtype=torch.bfloat16)
            packet = Fp8BlockwisePacket.allocate(m, h, aiter.dtypes.fp8, x.device)

            def collective(out, inp):
                with nccl.change_state(enable=True):
                    coordinator._all_gather_into_tensor(out, inp)

            def forward(
                x=x, norm_weight=norm_weight, residual=residual, packet=packet, m=m
            ):
                _, _, _, res = fused_rms_fp8_group_quant(
                    x,
                    norm_weight,
                    1e-6,
                    res1=residual,
                    dtype_quant=aiter.dtypes.fp8,
                    out_fp8=packet.quantized,
                    out_scale=packet.scales,
                )
                q, scales = all_gather_fp8_blockwise(packet, 4, collective)
                y = (
                    None
                    if m == 0
                    else aiter_w8a8_block_fp8_linear(
                        q, weight, [128, 128], weight_scale, input_scale=scales
                    )
                )
                return q, scales, y, res

            def validate(output, x=x, norm_weight=norm_weight, residual=residual, m=m):
                q, scales, y, res = output
                (local_q, local_s), _, _, local_res = fused_rms_fp8_group_quant(
                    x, norm_weight, 1e-6, res1=residual, dtype_quant=aiter.dtypes.fp8
                )
                q_ref = torch.empty((m * 4, h), device="cuda", dtype=torch.uint8)
                s_ref = torch.empty((m * 4, g), device="cuda", dtype=torch.float32)
                with nccl.change_state(enable=True):
                    nccl.all_gather(q_ref, local_q.view(torch.uint8))
                    nccl.all_gather(s_ref, local_s)
                torch.cuda.synchronize()
                assert torch.equal(q.view(torch.uint8), q_ref)
                torch.testing.assert_close(
                    scales.view(g, m * 4).t(), s_ref, rtol=0, atol=0
                )
                if residual is not None:
                    torch.testing.assert_close(res, local_res, rtol=0, atol=0)
                if m:
                    y_ref = aiter_w8a8_block_fp8_linear(
                        q_ref.view(aiter.dtypes.fp8),
                        weight,
                        [128, 128],
                        weight_scale,
                        input_scale=s_ref.t().contiguous().view(m * 4, g),
                    )
                    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)

            validate(forward())
            if min(row_counts) > 0:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with ca.capture():
                    with torch.cuda.stream(stream):
                        forward()
                    stream.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured = forward()
                for _ in range(3):
                    x.normal_()
                    if residual is not None:
                        residual.normal_()
                    graph.replay()
                    validate(captured)
            dist.barrier()
            if rank == 0:
                print(
                    f"PASS custom={custom}, rows={row_counts}: fused quant, one AG, scales, CK QKV, replay",
                    flush=True,
                )
    ca.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=8, join=True)

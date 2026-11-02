"""Real two-NPU FusedMoE parity, ND/NZ migration and independent forwards.

Run directly with SGLANG_DWDP_TEST_BASE_DEVICE selecting two consecutive NPUs.
"""

import os
import socket
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=90, suite="base-b-test-2-npu-a3")


class _Group:
    def __init__(self, rank, device_group):
        self.rank_in_group = rank
        self.world_size = 2
        self.cpu_group = dist.group.WORLD
        self.device_group = device_group

    def all_gather_object(self, value):
        values = [None, None]
        dist.all_gather_object(values, value, group=self.cpu_group)
        return values

    def barrier(self):
        dist.barrier(group=self.cpu_group)


def _worker(rank, port, base_device):
    import torch_npu

    from sglang.srt.hardware_backend.npu.utils import init_npu_backend
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
    from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
    from sglang.srt.runtime_context import (
        SpawnRanks,
        reset_context,
        set_global_dwdp_manager,
    )
    from sglang.srt.server_args import ServerArgs
    from sglang.test.parallel_groups import parallel_scope, publish

    device_id = base_device + rank
    torch.npu.set_device(device_id)
    torch.set_num_threads(1)
    init_npu_backend()
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    device_group = dist.new_group(backend="hccl")
    group = _Group(rank, device_group)
    server = ServerArgs(
        model_path="dummy",
        device="npu",
        tp_size=2,
        dwdp_size=2,
        ep_size=2,
        attn_dp_size=2,
        moe_dense_tp_size=1,
        enable_dp_lm_head=True,
        cuda_graph_config=CudaGraphConfig(decode=PhaseConfig(), prefill=PhaseConfig()),
        moe_runner_backend="ascend",
        disable_cuda_graph=True,
    )
    publish(server, role="test", ranks=SpawnRanks(world_rank=rank))
    from sglang.srt.hardware_backend.npu.dwdp.manager import NpuDwdpManager

    torch.set_default_device(f"npu:{device_id}")
    try:
        with parallel_scope(tp_group=group, moe_ep_group=group):
            for dtype in (torch.bfloat16, torch.float16, torch.int8):
                for nz_source in (False, True):
                    activation_dtype = torch.bfloat16 if dtype == torch.int8 else dtype
                    layers, references = [], []
                    widths = {
                        torch.bfloat16: (64, 96, 32, 128),
                        torch.float16: (64, 96, 32),
                        torch.int8: (64,),
                    }[dtype]
                    for li, width in enumerate(widths):
                        prefix = f"model.layers.{li}.mlp.experts"
                        quant_config = (
                            ModelSlimConfig(
                                {
                                    f"{prefix}.0.{proj}.weight": "W8A8_DYNAMIC"
                                    for proj in ("gate_proj", "up_proj", "down_proj")
                                }
                            )
                            if dtype == torch.int8
                            else None
                        )
                        options = dict(
                            num_experts=4,
                            hidden_size=128,
                            intermediate_size=width,
                            layer_id=li,
                            top_k=2,
                            params_dtype=activation_dtype,
                            with_bias=dtype != torch.int8,
                            quant_config=quant_config,
                            prefix=prefix,
                        )
                        local = FusedMoE(**options)
                        with parallel_scope(
                            moe_ep_size=1,
                            moe_ep_rank=0,
                            moe_dp_size=2,
                            moe_dp_rank=rank,
                            moe_ep_group=SimpleNamespace(world_size=1, rank_in_group=0),
                            moe_dp_group=group,
                        ):
                            reference = FusedMoE(**options)
                        names = ["w13_weight", "w2_weight"] + [
                            name for name, _ in reference.named_per_expert_tensors(4)
                        ]
                        for wi, name in enumerate(names):
                            value = getattr(reference, name)
                            full = (
                                torch.arange(value.numel(), device="cpu").remainder(29)
                                - 14
                            ).reshape(value.shape)
                            if name.endswith("_offset"):
                                full.zero_()
                            elif name.endswith("_scale"):
                                full = (full.abs() + 1).float() / 128
                            elif dtype != torch.int8:
                                full = full.float() / (128 + li * 16 + wi)
                            full = full.to(dtype=value.dtype, device=f"npu:{device_id}")
                            value.data.copy_(full)
                            if "bias" in name and dtype == torch.float16:
                                value.data = value.data.to(torch.float16)
                            shard = full[rank * 2 : (rank + 1) * 2].contiguous()
                            target = getattr(local, name)
                            target.data.copy_(shard)
                        if dtype == torch.int8:
                            for module in (local, reference):
                                module.quant_method.process_weights_after_loading(
                                    module
                                )
                        for name in ("w13_weight", "w2_weight"):
                            target = getattr(local, name)
                            target.data = torch_npu.npu_format_cast(
                                target.data, 29 if nz_source else 2
                            ).contiguous()
                            target = getattr(reference, name)
                            target.data = torch_npu.npu_format_cast(
                                target.data, 2
                            ).contiguous()
                        layer = torch.nn.Module()
                        layer.add_module("experts", local)
                        layers.append(layer)
                        references.append(reference)
                    model = torch.nn.Module()
                    model.layers = torch.nn.ModuleList(layers)
                    manager = NpuDwdpManager(server)
                    manager.setup(model)
                    set_global_dwdp_manager(manager)
                    # Rank 1 executes zero or one batches while rank 0 executes
                    # four, reading the idle rank's immutable source shards.
                    checks = []
                    batches = 4 if rank == 0 else int(nz_source)
                    for batch in range(batches):
                        tokens = 11 + rank * 8 + batch
                        x = (
                            torch.arange(tokens * 128, device="cpu")
                            .remainder(17)
                            .reshape(tokens, 128)
                        )
                        x = (
                            x.to(dtype=activation_dtype, device=f"npu:{device_id}")
                            / 128
                        )
                        ids = (
                            torch.arange(tokens * 2, device=f"npu:{device_id}")
                            .reshape(tokens, 2)
                            .remainder(4)
                            .int()
                        )
                        topk = StandardTopKOutput(
                            torch.full((tokens, 2), 0.5, dtype=activation_dtype),
                            ids,
                            None,
                        )
                        manager.prefetch_first_layers()
                        for li, layer in enumerate(layers):
                            local = layer.experts
                            actual = local(x, topk)
                            expected = references[li](x, topk)
                            checks.append((actual - expected).abs().float().max())
                    # No host synchronization was inserted between the forwards.
                    if checks:
                        torch.testing.assert_close(
                            torch.stack(checks).cpu(),
                            torch.zeros(len(checks), device="cpu"),
                            rtol=0,
                            atol=0,
                        )
                    manager.cleanup()
                    manager.cleanup()
                    set_global_dwdp_manager(None)
                    print(
                        f"PASS rank={rank} dtype={dtype} nz_source={nz_source} layers={len(layers)} batches={batches}",
                        flush=True,
                    )
    finally:
        reset_context()
        dist.destroy_process_group()


class TestNpuDwdp(unittest.TestCase):
    def test_two_rank_moe(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(
            _worker,
            args=(port, int(os.getenv("SGLANG_DWDP_TEST_BASE_DEVICE", "0"))),
            nprocs=2,
        )


if __name__ == "__main__":
    unittest.main()

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import sglang.srt.layers.moe.dwdp.rocm_vmm_manager as rocm_vmm_manager
from sglang.srt.layers.moe.dwdp.tensor_schema import DwdpTensorSchema
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, suite="stage-b-test-2-gpu-large-amd")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _FakeQuantMethod:
    def get_dwdp_tensor_schema(self, _layer):
        return DwdpTensorSchema(
            partitioned=(
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ),
        )

    def get_dwdp_tensor(self, layer, name):
        value = getattr(layer, name)
        return value.data if isinstance(value, nn.Parameter) else value


class _FakeExperts(nn.Module):
    def __init__(self, rank: int, world_size: int, layer_idx: int):
        super().__init__()
        self.num_global_routed_experts = world_size * 2
        self.num_fused_shared_experts = 1
        self.quant_method = _FakeQuantMethod()
        self.bound = False

        routed = torch.tensor(
            [rank * 2 + layer_idx * 100, rank * 2 + 1 + layer_idx * 100],
            dtype=torch.int32,
            device=f"cuda:{rank}",
        )
        shared = torch.tensor(
            [9000 + layer_idx],
            dtype=torch.int32,
            device=f"cuda:{rank}",
        )
        values = torch.cat([routed, shared])
        element_size = torch.tensor([], dtype=torch.int32).element_size()
        expert_bytes = int(os.environ.get("DWDP_TEST_EXPERT_BYTES", "32"))
        if expert_bytes % element_size != 0:
            raise ValueError("DWDP_TEST_EXPERT_BYTES must be int32-aligned")
        weight_width = expert_bytes // element_size
        self.w13_weight = nn.Parameter(
            values[:, None].repeat(1, weight_width), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            values[:, None].repeat(1, max(1, weight_width // 2)),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            values[:, None].repeat(1, 2), requires_grad=False
        )
        self.w2_weight_scale = nn.Parameter(values[:, None], requires_grad=False)

    def bind_full_expert_weights(self, weights):
        for name, tensor in weights.items():
            setattr(self, name, nn.Parameter(tensor, requires_grad=False))
        self.bound = True

    def unbind_dwdp_weights(self):
        self.bound = False

    def replace_expert_tensor(self, name, tensor):
        setattr(self, name, nn.Parameter(tensor, requires_grad=False))


class _FakeGroup:
    def __init__(self, device_group, cpu_group):
        self.device_group = device_group
        self.cpu_group = cpu_group

    def barrier(self):
        dist.barrier(group=self.cpu_group)


def _expected(world_size: int, layer_idx: int):
    values = [
        peer * 2 + offset + layer_idx * 100
        for peer in range(world_size)
        for offset in range(2)
    ]
    values.append(9000 + layer_idx)
    return values


def _vmm_worker(rank: int, world_size: int, port: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    fake_group = _FakeGroup(dist.group.WORLD, cpu_group)
    old_get_parallel = rocm_vmm_manager.get_parallel
    rocm_vmm_manager.get_parallel = lambda: SimpleNamespace(
        tp_rank=rank,
        tp_group=fake_group,
    )
    manager = None
    try:
        num_layers = int(os.environ.get("DWDP_TEST_NUM_LAYERS", "3"))
        experts = [
            _FakeExperts(rank, world_size, layer_idx) for layer_idx in range(num_layers)
        ]
        manager = rocm_vmm_manager.RocmVmmDwdpManager(
            SimpleNamespace(dwdp_size=world_size)
        )
        manager._collect_moe_layers = lambda _model: list(enumerate(experts))
        manager.setup(SimpleNamespace())
        assert all(layer.bound for layer in experts)

        manager.prefetch_first_layers()
        for layer_idx in (0, 1):
            manager.wait_prefetch(layer_idx)
            observed = experts[layer_idx].w13_weight[:, 0].cpu().tolist()
            assert observed == _expected(world_size, layer_idx)
            scales = experts[layer_idx].w13_weight_scale[:, 0].cpu().tolist()
            assert scales == _expected(world_size, layer_idx)

        if num_layers >= 3:
            manager.record_compute_and_prefetch_next(0)
            manager.wait_prefetch(2)
            assert experts[2].w2_weight[:, 0].cpu().tolist() == _expected(world_size, 2)
        dist.barrier(group=cpu_group)
    finally:
        if manager is not None:
            manager.cleanup()
        rocm_vmm_manager.get_parallel = old_get_parallel
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm DWDP VMM manager test")
def test_rocm_vmm_manager_full_expert_tensor():
    world_size = int(os.environ.get("DWDP_TEST_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} ROCm GPUs")
    mp.spawn(
        _vmm_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )

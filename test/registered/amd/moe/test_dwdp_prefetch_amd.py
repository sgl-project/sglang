import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import sglang.srt.layers.moe.dwdp.rocm_ipc as rocm_ipc
from sglang.srt.layers.moe.dwdp.tensor_schema import DwdpTensorSchema
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, suite="stage-c-test-large-8-gpu-amd-mi35x")


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
            replicated=("w13_weight_bias",),
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
        self.w13_weight = nn.Parameter(
            values[:, None].repeat(1, 8),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            values[:, None].repeat(1, 4),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            (values % 251).to(torch.uint8)[:, None].repeat(1, 2),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            (values % 251).to(torch.uint8)[:, None],
            requires_grad=False,
        )
        self.w13_weight_bias = nn.Parameter(
            values[:, None].to(torch.float32),
            requires_grad=False,
        )

    def bind_dwdp_partitioned_weights(self):
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


def _expected_partition(rank: int, layer_idx: int, is_last: bool):
    values = [rank * 2 + layer_idx * 100, rank * 2 + 1 + layer_idx * 100]
    if is_last:
        values.append(9000 + layer_idx)
    return values


def _prefetch_worker(rank: int, world_size: int, port: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    fake_group = _FakeGroup(dist.group.WORLD, cpu_group)
    old_get_parallel = rocm_ipc.get_parallel
    rocm_ipc.get_parallel = lambda: SimpleNamespace(
        tp_rank=rank,
        tp_group=fake_group,
    )
    try:
        experts = [_FakeExperts(rank, world_size, layer_idx) for layer_idx in range(4)]
        manager = rocm_ipc.RocmIpcDwdpManager(SimpleNamespace(dwdp_size=world_size))
        manager._collect_moe_layers = lambda _model: list(enumerate(experts))
        manager.setup(SimpleNamespace())
        assert all(layer.bound for layer in experts)

        manager.prefetch_first_layers()
        for layer_idx in (0, 1):
            manager.wait_prefetch(layer_idx)
            view = manager.get_partition_view(layer_idx, "w13_weight")
            assert view.partition_sizes == tuple(
                3 if peer == world_size - 1 else 2 for peer in range(world_size)
            )
            for peer, partition in enumerate(view.tensors):
                observed = partition[:, 0].cpu().tolist()
                assert observed == _expected_partition(
                    peer,
                    layer_idx,
                    peer == world_size - 1,
                )

        manager.record_compute_and_prefetch_next(0)
        manager.wait_prefetch(2)
        view = manager.get_partition_view(2, "w2_weight")
        for peer, partition in enumerate(view.tensors):
            assert partition[:, 0].cpu().tolist() == _expected_partition(
                peer,
                2,
                peer == world_size - 1,
            )
        manager.cleanup()
        assert all(not layer.bound for layer in experts)

        manager.setup(SimpleNamespace())
        assert all(layer.bound for layer in experts)
        manager.prefetch_layer(0)
        manager.wait_prefetch(0)
        view = manager.get_partition_view(0, "w13_weight")
        for peer, partition in enumerate(view.tensors):
            assert partition[:, 0].cpu().tolist() == _expected_partition(
                peer,
                0,
                peer == world_size - 1,
            )
        manager.cleanup()
        dist.barrier(group=cpu_group)
    finally:
        rocm_ipc.get_parallel = old_get_parallel
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm DWDP prefetch test")
def test_rocm_ipc_double_buffer_prefetch():
    world_size = int(os.environ.get("DWDP_TEST_WORLD_SIZE", "2"))
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} ROCm GPUs")
    mp.spawn(
        _prefetch_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )

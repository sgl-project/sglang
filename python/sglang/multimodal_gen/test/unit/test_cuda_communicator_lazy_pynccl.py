# SPDX-License-Identifier: Apache-2.0
"""A group's PyNccl communicator is built on its first all-reduce, not with the group."""

import torch

from sglang.multimodal_gen.runtime.distributed.device_communicators import (
    base_device_communicator,
    cuda_communicator,
    pynccl,
)


class _FakePyNccl:
    made: list = []
    disabled = False

    def __init__(self, group, device):
        _FakePyNccl.made.append(group)

    def all_reduce(self, tensor, op=None):
        return tensor * 2


def _communicator(monkeypatch):
    def fake_base_init(self, cpu_group, device=None, device_group=None, unique_name=""):
        self.cpu_group, self.device, self.device_group = cpu_group, device, device_group
        self.world_size, self.rank_in_group = 2, 0

    monkeypatch.setattr(
        base_device_communicator.DeviceCommunicatorBase, "__init__", fake_base_init
    )
    monkeypatch.setattr(pynccl, "PyNcclCommunicator", _FakePyNccl)
    _FakePyNccl.made = []
    return cuda_communicator.CudaCommunicator(
        cpu_group="cpu", device=torch.device("cpu")
    )


def test_pynccl_is_created_on_the_first_all_reduce(monkeypatch):
    comm = _communicator(monkeypatch)
    assert _FakePyNccl.made == []
    out = comm.all_reduce(torch.ones(2))
    assert torch.equal(out, torch.full((2,), 2.0))
    comm.all_reduce(torch.ones(2))
    assert _FakePyNccl.made == ["cpu"]


def test_point_to_point_never_creates_it(monkeypatch):
    comm = _communicator(monkeypatch)
    sent = []
    monkeypatch.setattr(
        torch.distributed, "send", lambda tensor, dst, group: sent.append(dst)
    )
    comm.ranks = [0, 1]
    comm.send(torch.ones(1))
    assert sent == [1] and _FakePyNccl.made == []

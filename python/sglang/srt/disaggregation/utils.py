from __future__ import annotations

import os
from collections import deque
from enum import Enum
from typing import List

import numpy as np
import pynvml
import pyverbs.device as d
import torch
import torch.distributed as dist


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def poll_and_all_reduce(pollers, gloo_group):
    polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self) -> List[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    PYVERBS = "pyverbs"
    FAKE = "fake"


class KVClassType(Enum):
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(transfer_backend: TransferBackend, class_type: KVClassType):
    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: MooncakeKVReceiver,
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.PYVERBS:
        from sglang.srt.disaggregation.pyverbs import (
            PyverbsKVBootstrapServer,
            PyverbsKVManager,
            PyverbsKVReceiver,
            PyverbsKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER: PyverbsKVManager,
            KVClassType.SENDER: PyverbsKVSender,
            KVClassType.RECEIVER: PyverbsKVReceiver,
            KVClassType.BOOTSTRAP_SERVER: PyverbsKVBootstrapServer,
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def groups_kv_indices_continuity_by_numpy(arr1):
    arr1 = np.array(arr1)

    # Find discontinuous index points (where adjacent elements differ by more than 1)
    split_indices = np.where(np.diff(arr1) != 1)[0] + 1

    # Split array using split_indices
    grouped_arr1 = np.split(arr1, split_indices)

    return [list(g) for g in grouped_arr1]


def get_device_list(prefix, gpu_no=0, roce_version=2, port_num=1):
    lst = d.get_device_list()
    if len(lst) == 0:
        print("No IB devices")
        return []
    device_list = {}
    for dev in lst:
        if dev.name.decode().startswith(prefix):
            with d.Context(name=dev.name.decode()) as ctx:
                gid_tbl_len = ctx.query_port(port_num).gid_tbl_len
                if gid_tbl_len > 0:
                    ctx.query_gid(port_num=port_num, index=roce_version)
                    # Get PCI address from sysfs
                    dev_path = f"/sys/class/infiniband/{dev.name.decode()}/device"
                    if os.path.exists(dev_path):
                        pci_addr = os.readlink(dev_path).split("/")[
                            -1
                        ]  # Format like "0000:19:00.0"
                    device_list[dev.name.decode()] = pci_addr

    return device_list


def get_gpu_pci_address(gpu_no):
    """Get PCI address of specified GPU device"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_no)
    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
    pynvml.nvmlShutdown()
    return pci_info.busId


def get_net_device_from_rdma(rdma_dev):
    """Get network interface name corresponding to RoCE device"""
    net_path = f"/sys/class/infiniband/{rdma_dev}/device/net"
    if os.path.exists(net_path):
        return os.listdir(net_path)[0]  # Read network interface name
    return None


def normalize_pci_addr(pci_addr):
    """Standardize PCI address format, e.g. 00000000:08:00.0 -> 0000:08:00.0"""
    parts = pci_addr.split(":")
    if len(parts) == 3:  # Format like "00000000:08:00.0"
        return f"{int(parts[0], 16):04x}:{parts[1]}:{parts[2]}"  # Convert to "0000:08:00.0"
    return pci_addr  # Return original format


def find_best_rdma_ib_device(gpu_no, prefix="mlx"):
    """Find the most affinity RoCE network card based on GPU device number"""
    gpu_pci = normalize_pci_addr(get_gpu_pci_address(gpu_no))
    roce_devices = {
        k: normalize_pci_addr(v) for k, v in get_device_list(prefix).items()
    }

    best_rdma_dev = None
    min_distance = float("inf")

    for rdma_dev, rdma_pci in roce_devices.items():
        if rdma_pci[:5] == gpu_pci[:5]:  # Ensure same NUMA node
            distance = abs(
                int(rdma_pci.split(":")[1], 16) - int(gpu_pci.split(":")[1], 16)
            )
            if distance < min_distance:
                min_distance = distance
                best_rdma_dev = rdma_dev

    if best_rdma_dev:
        net_dev = get_net_device_from_rdma(best_rdma_dev)
        return best_rdma_dev, net_dev


if __name__ == "__main__":
    gpu_no = 0  # GPU device number to query
    rdma_dev, net_dev = find_best_rdma_ib_device(gpu_no)
    print(
        f"GPU {gpu_no} most affinity RDMA device: {rdma_dev}, corresponding network interface: {net_dev}"
    )

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


FakeBootstrapHost = "2.2.2.2"


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
    NIXL = "nixl"
    FAKE = "fake"


class KVClassType(Enum):
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(transfer_backend: TransferBackend, class_type: KVClassType):
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

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
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    if transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    if transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

        class_mapping = {
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaruanteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


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
    """Get PCI address for specified GPU device"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_no)
    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
    pynvml.nvmlShutdown()
    return pci_info.busId  #


def get_net_device_from_rdma(rdma_dev):
    """Get network interface name corresponding to RoCE device"""
    net_path = f"/sys/class/infiniband/{rdma_dev}/device/net"
    if os.path.exists(net_path):
        return os.listdir(net_path)[0]  # Read network interface name
    return None


def normalize_pci_addr(pci_addr):
    """Normalize PCI address format, e.g. 00000000:08:00.0 -> 0000:08:00.0"""
    parts = pci_addr.split(":")
    if len(parts) == 3:  # Format like "00000000:08:00.0"
        return f"{int(parts[0], 16):04x}:{parts[1]}:{parts[2]}"  # Convert to "0000:08:00.0"
    return pci_addr  # Return original format


def find_best_rdma_ndevices_for_gpu(gpu_no, prefix="", n=1):
    """Find the n closest RoCE network cards based on GPU device number"""
    gpu_pci = normalize_pci_addr(get_gpu_pci_address(gpu_no))
    roce_devices = {
        k: normalize_pci_addr(v) for k, v in get_device_list(prefix).items()
    }

    # List to store (distance, rdma_dev) pairs
    device_distances = []

    for rdma_dev, rdma_pci in roce_devices.items():
        if rdma_pci[:5] == gpu_pci[:5]:  # **Ensure same NUMA node**
            distance = abs(
                int(rdma_pci.split(":")[1], 16) - int(gpu_pci.split(":")[1], 16)
            )
            device_distances.append((distance, rdma_dev))

    # Sort by distance and take top n
    device_distances.sort()  # Sort by distance (first element of tuple)
    closest_devices = device_distances[:n]

    # Get network interfaces for the closest devices
    result = []
    for _, rdma_dev in closest_devices:
        net_dev = get_net_device_from_rdma(rdma_dev)
        if net_dev:
            result.append((rdma_dev, net_dev))
    return result


def get_rdma_devices_by_gpu_no(gpu_no, prefix="", n=1):
    """Get the closet RDMA devices for the specified GPU device number"""
    closest_devices = find_best_rdma_ndevices_for_gpu(gpu_no, prefix, n)
    return ",".join([x[0] for x in closest_devices])


if __name__ == "__main__":
    gpu_no = 0  # GPU device number to query
    n = 2  # Number of closest devices to find
    closest_devices = find_best_rdma_ndevices_for_gpu(gpu_no, n=n)
    print(",".join([x[0] for x in closest_devices]))
    for i, (rdma_dev, net_dev) in enumerate(closest_devices):
        print(
            f"#{i+1} closest RDMA device for GPU {gpu_no}: {rdma_dev}, corresponding network interface: {net_dev}"
        )

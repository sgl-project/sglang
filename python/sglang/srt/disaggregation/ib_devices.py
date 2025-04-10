#!/usr/bin/env python
# coding:utf-8
"""
@author: nivic ybyang7
@license: Apache Licence
@file: ib_devices
@time: 2025/04/03
@contact: ybyang7@iflytek.com
@site:
@software: PyCharm

# Code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  God Bless   ┣┓
                ┃  No BUG!     ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import os

#  Copyright (c) 2022. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import pyverbs.device as d
import pynvml

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
                        pci_addr = os.readlink(dev_path).split("/")[-1]  # Format like "0000:19:00.0"
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

def find_best_roce_for_gpu(gpu_no, prefix="mlx"):
    """Find the most affinity RoCE network card based on GPU device number"""
    gpu_pci = normalize_pci_addr(get_gpu_pci_address(gpu_no))
    roce_devices = {k: normalize_pci_addr(v) for k, v in get_device_list(prefix).items()}

    best_rdma_dev = None
    min_distance = float("inf")

    for rdma_dev, rdma_pci in roce_devices.items():
        if rdma_pci[:5] == gpu_pci[:5]:  # Ensure same NUMA node
            distance = abs(int(rdma_pci.split(":")[1], 16) - int(gpu_pci.split(":")[1], 16))
            if distance < min_distance:
                min_distance = distance
                best_rdma_dev = rdma_dev

    if best_rdma_dev:
        net_dev = get_net_device_from_rdma(best_rdma_dev)
        return best_rdma_dev, net_dev

if __name__ == '__main__':
    gpu_no = 0  # GPU device number to query
    rdma_dev, net_dev = find_best_roce_for_gpu(gpu_no)
    print(f"GPU {gpu_no} most affinity RDMA device: {rdma_dev}, corresponding network interface: {net_dev}")

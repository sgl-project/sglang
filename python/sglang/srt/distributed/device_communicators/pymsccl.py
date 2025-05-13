# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/pynccl.py

import os
import ctypes
import logging
from contextlib import contextmanager
from typing import Union, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
from sglang.srt.distributed.utils import StatelessProcessGroup

import cupy as cp
import numpy as np
import netifaces as ni
import ipaddress
import mscclpp.comm as mscclpp_comm
from mscclpp.utils import KernelBuilder, pack
from mscclpp import Transport, MemoryChannel, MemoryDevice2DeviceSemaphore, ProxyService

logger = logging.getLogger(__name__)

def find_best_config(randn_input, mscclpp_algo, niter=20):
    logger.info(f"[{dist.get_rank()}] MSCCL starts to find best config")
    best_time, best_config = 10000000000.0, None
    stream = torch.cuda.Stream()
    mscclpp_algo.register_tensor(randn_input)
    for config in mscclpp_algo.auto_tune():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(niter):
                mscclpp_algo(stream)
        for _ in range(5):
            graph.replay()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(niter):
            graph.replay()
        end_event.record()
        end_event.synchronize()
        cur_time = start_event.elapsed_time(end_event)
        if cur_time < best_time:
            best_time, best_config = cur_time, config
    return best_config

def register_device_memory_handles(
    group: mscclpp_comm.CommGroup, 
    connections, 
    semaphores, 
    registered_memories, 
    tensor_data_ptr, 
    scratch_data_ptr,
    device_handles_cp: Optional[cp.ndarray]
):
    memory_channels = {}
    for rank in connections:
        memory_channels[rank] = MemoryChannel(
            semaphores[rank], registered_memories[rank], tensor_data_ptr, scratch_data_ptr
        )
    device_handles = []
    for rank in range(group.nranks):
        if rank != group.my_rank and rank in memory_channels:
            device_handles.append(memory_channels[rank].device_handle().raw)
    if device_handles_cp is None:
        return cp.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8)
    np_array = np.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8)
    cp.cuda.runtime.memcpy(device_handles_cp.data.ptr, np_array.ctypes.data, np_array.nbytes, cp.cuda.runtime.memcpyHostToDevice)

class MscclppAllReduce1Shot1NodeLL:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        max_bytes: int,
        device: torch.device,
        block_size: int = 512,
        nblocks: int = 21,
    ):
        self.group = group
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)

        self.group.barrier()
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, Transport.CudaIpc)
        
        # create a memory_channel for each remote neighbor
        self.scratch = torch.empty(max_bytes // torch.float32.itemsize * 8, dtype=torch.float32, device=device)
        self.semaphores = self.group.make_semaphore(self.connections, MemoryDevice2DeviceSemaphore)
        self.registered_memories = self.group.register_tensor_with_connections(self.scratch, self.connections)
        self.device_handles_cp = register_device_memory_handles(
            self.group, self.connections, self.semaphores, self.registered_memories, 
            self.scratch.data_ptr(), self.scratch.data_ptr(), None
        )
        self.memory = None
        self.memory_out = None
        self.params = None
        
        self.dtype2str = {
            torch.float16: "__half",
            torch.bfloat16: "__nv_bfloat16",
            torch.float: "float",
            torch.int32: "int",
        }
        
        self.msg_sz2param = {}
        
        # build kernel
        self.nblocks = nblocks
        self.block_size = block_size
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.dtype2kernel = {}
        self.kernel = None
        for dtype, dtype_str in self.dtype2str.items():
            self.dtype2kernel[dtype] = KernelBuilder(
                file="msccl_allreduce.cu", 
                kernel_name="allreduce_1shot_1node", 
                file_dir=file_dir, 
                macro_dict={"TYPE": dtype_str}
            ).get_compiled_kernel()
        
    def register_tensor(self, tensor: torch.Tensor):
        self.memory = tensor
        self.memory_out = torch.empty_like(self.memory)
        register_device_memory_handles(
            self.group, self.connections, self.semaphores, self.registered_memories, 
            tensor.data_ptr(), self.scratch.data_ptr(), self.device_handles_cp
        )
        if (tensor.dtype, tensor.numel()) in self.msg_sz2param:
            self.nblocks, self.block_size = self.msg_sz2param[(tensor.dtype, tensor.numel())]
        self.set_params()
        self.kernel = self.dtype2kernel[tensor.dtype]
        
    def __call__(self, stream):
        self.kernel.launch_kernel(self.params, self.nblocks, self.block_size, 0, stream)
        return self.memory_out

    def set_params(self, nblocks: int = None, block_size: int = None):
        if nblocks is not None:
            self.nblocks = nblocks
        if block_size is not None:
            self.block_size = block_size

        self.params = b""
        self.params += pack(
            self.device_handles_cp,
            self.memory,
            self.scratch,
            self.memory_out,
            self.group.my_rank,
            self.group.nranks,
            ctypes.c_size_t(self.memory.numel()),
        )

    def auto_tune(self):
        nblocks_to_try = [21, 42, 63, 84, 105]
        block_size_to_try = [256, 512, 1024]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                self.set_params(nblocks, block_size)
                yield nblocks, block_size

class MscclppAllReduce1Shot2Nodes:
    def __init__(
        self,
        group: mscclpp_comm.CommGroup,
        max_bytes: int,
        device: torch.device,
        nranks_per_node: int,
        proxy_service: ProxyService,
        nblocks: int = 21,
        block_size: int = 512,
    ):
        IB_TRANSPORTS = [
            Transport.IB0,
            Transport.IB1,
            Transport.IB2,
            Transport.IB3,
            Transport.IB4,
            Transport.IB5,
            Transport.IB6,
            Transport.IB7,
        ]
        self.dtype2str = {
            torch.float16: "__half",
            torch.bfloat16: "__nv_bfloat16",
            torch.float: "float",
            torch.int32: "int",
        }
        self.group = group
        self.nranks_per_node = nranks_per_node
        self.in_same_node = lambda rank: rank // nranks_per_node == self.group.my_rank // nranks_per_node
        remote_nghrs = list(range(self.group.nranks))
        remote_nghrs.remove(self.group.my_rank)
        transports = {}
        for rank in remote_nghrs:
            if self.in_same_node(rank):
                transports[rank] = Transport.CudaIpc
            else:
                transports[rank] = IB_TRANSPORTS[rank % nranks_per_node]
        self.group.barrier()
        
        # create a connection for each remote neighbor
        self.connections = self.group.make_connection(remote_nghrs, transports)
        self.proxy_service = proxy_service
        self.scratch = torch.empty(max_bytes // torch.float32.itemsize * 8, dtype=torch.float32, device=device)
        self.put_buff = torch.empty(max_bytes // torch.float32.itemsize * 8 // nranks_per_node, dtype=torch.float32, device=device)
        self.same_node_connections = {rank: conn for rank, conn in self.connections.items() if self.in_same_node(rank)}
        self.across_node_connections = {rank: conn for rank, conn in self.connections.items() if not self.in_same_node(rank)}
        
        # create a memory_channel for each remote neighbor
        self.memory_semaphores = self.group.make_semaphore(self.same_node_connections, MemoryDevice2DeviceSemaphore)
        self.memory_registered_memories = self.group.register_tensor_with_connections(self.scratch, self.same_node_connections)
        self.mem_device_handles_cp = register_device_memory_handles(
            self.group, self.same_node_connections, self.memory_semaphores, self.memory_registered_memories, 
            self.scratch.data_ptr(), self.scratch.data_ptr(), None
        )
        
        port_channels = self.group.make_port_channels_with_scratch(
            self.proxy_service, self.put_buff, self.scratch, self.across_node_connections
        )
        proxy_device_handles = []
        for rank in range(self.group.nranks):
            if rank != self.group.my_rank and not self.in_same_node(rank):
                proxy_device_handles.append(port_channels[rank].device_handle().raw)
        self.proxy_device_handles_cp = cp.asarray(memoryview(b"".join(proxy_device_handles)), dtype=cp.uint8)

        self.msg_sz2param = {}
        
        # build kernel
        self.nblocks = nblocks
        self.block_size = block_size
        file_dir = os.path.dirname(os.path.abspath(__file__))
        self.dtype2kernel = {}
        self.kernel = None
        for dtype, dtype_str in self.dtype2str.items():
            self.dtype2kernel[dtype] = KernelBuilder(
                file="msccl_allreduce.cu", 
                kernel_name="allreduce5", 
                file_dir=file_dir, 
                macro_dict={"TYPE": dtype_str}
            ).get_compiled_kernel()
    
    def register_tensor(self, tensor: torch.Tensor):
        self.memory = tensor
        self.memory_out = torch.empty_like(self.memory)
        register_device_memory_handles(
            self.group, self.same_node_connections, self.memory_semaphores, self.memory_registered_memories, 
            tensor.data_ptr(), self.scratch.data_ptr(), self.mem_device_handles_cp
        )
        if (tensor.dtype, tensor.numel()) in self.msg_sz2param:
            self.nblocks, self.block_size = self.msg_sz2param[(tensor.dtype, tensor.numel())]
        self.set_params()
        self.kernel = self.dtype2kernel[tensor.dtype]

    def __call__(self, stream):
        self.kernel.launch_kernel(self.params, self.nblocks, self.block_size, 0, stream)
        return self.memory_out

    def set_params(self, nblocks: int = None, block_size: int = None):
        if nblocks is not None:
            self.nblocks = nblocks
        if block_size is not None:
            self.block_size = block_size

        self.params = b""
        self.params += pack(
            self.mem_device_handles_cp,
            self.proxy_device_handles_cp,
            self.memory,
            self.scratch,
            self.put_buff,
            self.memory_out,
            self.group.my_rank,
            self.nranks_per_node,
            self.group.nranks,
            bytes(4),  # padding for memory alignment
            ctypes.c_size_t(self.memory.numel()),
        )

    def auto_tune(self):
        nblocks_to_try = [21, 42, 84]
        block_size_to_try = [256, 512, 1024]
        for nblocks in nblocks_to_try:
            for block_size in block_size_to_try:
                self.set_params(nblocks, block_size)
                yield nblocks, block_size

def is_valid(ip):
    """
    Check if the IP address is valid for connecting to other devices.
    This excludes loopback (127.0.0.1) and link-local (169.254.x.x) addresses.
    """
    ip_obj = ipaddress.ip_address(ip)
    return not (ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast)

def get_netinterface_info():
    """
    Returns the name of the first network interface with a valid IP address that it finds.
    """
    interfaces = ni.interfaces()
    for interface in interfaces:
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for addr in addresses[ni.AF_INET]:
                ip_address = addr["addr"]
                if is_valid(ip_address):
                    logger.info(f"MSCCL Selected Interface: {interface}, IP Address: {ip_address}")
                    return interface, ip_address
    return None, None

class PyMscclCommunicator:
    def __init__(
        self, 
        cpu_group: Union[ProcessGroup, StatelessProcessGroup],
        device: torch.device,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.rank = dist.get_rank(cpu_group)
        self.world_size = dist.get_world_size(cpu_group)
        network_interface, my_ip = get_netinterface_info()
        assert network_interface is not None, "No network interface found"
        objects = [my_ip] if self.rank == 0 else [None]
        dist.broadcast_object_list(objects, src=0, device=torch.device("cpu"), group=cpu_group)
        ifIpPortTrio = network_interface + ":" + objects[0] + f":{50000 + self.world_size}"  # some random port
        mscclpp_group = mscclpp_comm.CommGroup(
            interfaceIpPortTrio=ifIpPortTrio, rank=self.rank, size=self.world_size
        )
        self.MAX_BYTES = 2**20
        if self.world_size == torch.cuda.device_count():
            self.allreduce_algo = MscclppAllReduce1Shot1NodeLL(mscclpp_group, self.MAX_BYTES, device)
        elif self.world_size == torch.cuda.device_count() * 2:
            proxy_service = ProxyService()
            self.allreduce_algo = MscclppAllReduce1Shot2Nodes(mscclpp_group, self.MAX_BYTES, device, torch.cuda.device_count(), proxy_service)
            proxy_service.start_proxy()
        else:
            raise ValueError(f"world size {self.world_size} is not supported")

        self.disabled = False
        self._IS_CAPTURING = False
        self.stream = torch.cuda.Stream()
        self.graph_input_set = set()
    
    def post_process_graph_input(self) -> bool:
        for tensor_dtype, tensor_numel in self.graph_input_set:
            if (tensor_dtype, tensor_numel) not in self.allreduce_algo.msg_sz2param:
                randn_input = torch.randn((tensor_numel, ), dtype=tensor_dtype, device=self.device)
                best_config = find_best_config(randn_input, self.allreduce_algo)
                objects = [best_config] if self.rank == 0 else [None]
                dist.broadcast_object_list(objects, src=0, device=torch.device("cpu"), group=self.cpu_group)
                best_config = objects[0]
                self.allreduce_algo.msg_sz2param[(tensor_dtype, tensor_numel)] = best_config
        
    def should_msccl_allreduce(self, inp: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> bool:
        if self.disable or self.allreduce_algo is None:
            return False
        if inp.dtype not in self.allreduce_algo.dtype2kernel:
            return False
        # only support sum op
        if op != ReduceOp.SUM:
            return False
        if inp.numel() * inp.element_size() > self.MAX_BYTES:
            return False
        return True

    def all_reduce(
        self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM
    ):
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                self.graph_input_set.add((tensor.dtype, tensor.numel()))
        stream = torch.cuda.current_stream()
        self.allreduce_algo.register_tensor(tensor)
        return self.allreduce_algo(stream)
    
    @contextmanager
    def change_state(
        self, enable: Optional[bool] = None,
    ):
        if enable is None:
            # guess a default value when not specified
            enable = self.available
        
        old_disable = self.disabled
        self.disabled = not enable
        
        yield
        
        self.disabled = old_disable
        if enable:
            self.post_process_graph_input()
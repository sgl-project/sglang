from __future__ import annotations

import dataclasses
import os
import random
import threading
import warnings
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple
from abc import ABC
from functools import lru_cache, wraps

import numpy as np
import requests
import torch
import torch.distributed as dist
import time

from sglang.srt.utils import get_ip, get_local_ip_by_remote

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

FakeBootstrapHost = "2.2.2.2"

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def poll_and_all_reduce(pollers, gloo_group):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
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
    MOONCAKE_ASYNC = "mooncake_async"
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
    if transfer_backend == TransferBackend.MOONCAKE_ASYNC:
        from sglang.srt.disaggregation.mooncake import (
            MooncakeAsyncKVManager,
            MooncakeAsyncKVReceiver,
            MooncakeAsyncKVSender,
            MooncakeKVBootstrapServer,
        )

        class_mapping = {
            KVClassType.MANAGER: MooncakeAsyncKVManager,
            KVClassType.SENDER: MooncakeAsyncKVSender,
            KVClassType.RECEIVER: (MooncakeAsyncKVReceiver),
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
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


@dataclasses.dataclass
class PDRegistryRequest:
    """A request to register a machine itself to the LB."""

    mode: str
    registry_url: str
    bootstrap_port: Optional[int] = None

    def __post_init__(self):
        if self.mode == "prefill" and self.bootstrap_port is None:
            raise ValueError("Bootstrap port must be set in PREFILL mode.")
        elif self.mode == "decode" and self.bootstrap_port is not None:
            raise ValueError("Bootstrap port must not be set in DECODE mode.")
        elif self.mode not in ["prefill", "decode"]:
            raise ValueError(
                f"Invalid mode: {self.mode}. Must be 'prefill' or 'decode'."
            )


def register_disaggregation_server(
    mode: str, server_port: int, bootstrap_port: int, pdlb_url: str
):
    boostrap_port = bootstrap_port if mode == "prefill" else None
    registry_request = PDRegistryRequest(
        mode=mode,
        registry_url=f"http://{get_ip()}:{server_port}",
        bootstrap_port=boostrap_port,
    )
    res = requests.post(
        f"{pdlb_url}/register",
        json=dataclasses.asdict(registry_request),
    )
    if res.status_code != 200:
        warnings.warn(
            f"Failed to register disaggregation server: {res.status_code} {res.text}"
        )


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []


class MetadataBuffers:
    def __init__(self, size: int, max_top_logprobs_num: int = 128):
        # TODO: abort top_logprobs_num > 128 in PD

        # We transfer the metadata of first output token to decode
        # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
        self.output_ids = torch.zeros((size, 16), dtype=torch.int32, device="cpu")
        self.output_token_logprobs_val = torch.zeros(
            (size, 16), dtype=torch.float32, device="cpu"
        )
        self.output_token_logprobs_idx = torch.zeros(
            (size, 16), dtype=torch.int32, device="cpu"
        )
        self.output_top_logprobs_val = torch.zeros(
            (size, max_top_logprobs_num), dtype=torch.float32, device="cpu"
        )
        self.output_top_logprobs_idx = torch.zeros(
            (size, max_top_logprobs_num), dtype=torch.int32, device="cpu"
        )

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
        )

    def set_buf(self, req: Req):

        self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
        if req.return_logprob:
            if req.output_token_logprobs_val:  # not none or empty list
                self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_val[0]
                )
            if req.output_token_logprobs_idx:  # not none or empty list
                self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                    req.output_token_logprobs_idx[0]
                )

            if req.output_top_logprobs_val:  # not none or empty list
                self.output_top_logprobs_val[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_val[0])
                ] = torch.tensor(
                    req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                )
            if req.output_top_logprobs_idx:  # not none or empty list
                self.output_top_logprobs_idx[req.metadata_buffer_index][
                    : len(req.output_top_logprobs_idx[0])
                ] = torch.tensor(
                    req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                )


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()
def np_cache(function):
    @lru_cache(maxsize=1024)
    def cached_wrapper(*hashable_arrays):
        np_array = (np.array(x) for x in hashable_arrays)
        return function(*np_array)

    @wraps(function)
    def wrapper(*arrays):
        return cached_wrapper(*(tuple(x) for x in arrays))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups

@np_cache
def cached_group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int64], dst_indices: npt.NDArray[np.int64]
) -> Tuple[List[npt.NDArray[np.int64]], List[npt.NDArray[np.int64]]]:
    return group_concurrent_contiguous(src_indices, dst_indices)

class StreamAsyncSubmitter(ABC):
    COUNT_NUM_MAX = 2**62
    """a class to get cuda stream status async and submit jobs after cuda kernel launched"""

    def __init__(self, submit_func: Callable):
        self._submit_func = submit_func
        # the step count means the how many layers been call with `step_async`
        self._step_count = 0
        # the sent_count means the how many layers is been actually sent
        self._sent_count = 0
        # the init value of finished_layer
        self._finished_layer_init = torch.zeros([1], dtype=torch.int64).cuda()
        # the counter in cuda
        self._finished_layer_cuda = torch.zeros([1], dtype=torch.int64).cuda()
        # the one in cuda
        self._one_cuda = torch.ones([1], dtype=torch.int64).cuda()
        # the cpu version of finished_layer cuda
        self._finished_layer_cpu = torch.zeros([1], dtype=torch.int64, pin_memory=True)
        self._lock = threading.Lock()

    def flush_step(self):
        with self._lock:
            current_finished_layer = int(self._finished_layer_cpu[0])
            # because the current_finished_layer may be larger than step_count,
            # so we need to calculate the real count
            submit_count = (
                current_finished_layer + self.COUNT_NUM_MAX - self._sent_count
            ) % self.COUNT_NUM_MAX
            for _ in range(submit_count):
                self._submit_func()
            self._sent_count = current_finished_layer

    def get_sent_count(self):
        return self._sent_count

    def get_step_count(self):
        return self._step_count

    def step_async(self):
        # we using a non blocking copy to sync the cuda and cpu
        # for example, if we call 5 times of step_async, and 3 layers are finished,
        # the step_count==5, the finished_layer_cuda==3
        # and we get the finished_layer_cpu would be 3
        # if the sent_count==2, in flush_step, we just call self._submit_func once
        # we need to copy the finished_layer_init to finished_layer_cuda, if the step_count if almost equal to COUNT_NUM_MAX
        if self._step_count == self.COUNT_NUM_MAX - 1:
            self._finished_layer_cuda.copy_(
                self._finished_layer_init, non_blocking=True
            )
        else:
            self._finished_layer_cuda.add_(self._one_cuda)
        self._finished_layer_cpu.copy_(self._finished_layer_cuda, non_blocking=True)
        # and we need to update the step_count
        self._step_count = (self._step_count + 1) % self.COUNT_NUM_MAX
        self.flush_step()

    def wait_sent_finish(self, task_stop_count):
        # because COUNT_NUM_MAX is very large, we can make sure that if diff is > COUNT_NUM_MAX / 2 means the flush is finished
        # and if the current_sent_count == task_stop_count also means the flush is not finished
        # so if current_sent_count != task_stop_count and diff < COUNT_NUM_MAX / 2, the flush is not finished
        task_stop_count = task_stop_count % self.COUNT_NUM_MAX
        current_sent_count = self.get_sent_count()
        while (task_stop_count != current_sent_count) and (
            (task_stop_count + self.COUNT_NUM_MAX - current_sent_count)
            % self.COUNT_NUM_MAX
        ) < self.COUNT_NUM_MAX / 2:
            time.sleep(1e-3)
            self.flush_step()
            current_sent_count = self.get_sent_count()

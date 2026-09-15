import json
import logging
import os
import queue
import threading
import time
from enum import IntEnum
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import yaml

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool as NSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool_host import synchronized
from sglang.srt.runtime_context import get_parallel

try:
    import eic
except ImportError:
    eic = None

logger = logging.getLogger(__name__)

REMOTE_EIC_YAML_ENV_VAR = "REMOTE_EIC_YAML"

# GDR Bounce Buffer
G_GDRBounceBufferSize = 1 * 1024 * 1024 * 1024
G_GDRBounceTensorCount = 128  # calculated by G_GDRBounceBufferSize / kvcache_size

# gpu direct rdma for kv set
G_EnableKVSetGPUDirect = False

# gpu direct rdma for kv get
G_EnableKVGetGPUDirect = True

# gpu nic affinity
G_EnableGPUNicAffinity = False

# async kv set
G_EnableAsyncKVSet = False
G_KVSetTTLOption = -1

# The local EIC agent used by DSv4 commonly has a 4 MiB max value class.  A
# packed DSv4 radix page is much larger, so DSv4 splits one logical page across
# multiple EIC keys and keeps the radix/page hit test atomic.
DEFAULT_DSV4_EIC_CHUNK_BYTES = 4 * 1024 * 1024


# default H20 gpu nic affinity
GPUNicAffinity = {
    "cuda:0": "eth1",
    "cuda:1": "eth1",
    "cuda:2": "eth2",
    "cuda:3": "eth2",
    "cuda:4": "eth3",
    "cuda:5": "eth3",
    "cuda:6": "eth4",
    "cuda:7": "eth4",
}

# default H20 cpu nic affinity
CPUNicAffinity = {
    "cuda:0": "cpu",
    "cuda:1": "cpu",
    "cuda:2": "cpu",
    "cuda:3": "cpu",
    "cuda:4": "cpu",
    "cuda:5": "cpu",
    "cuda:6": "cpu",
    "cuda:7": "cpu",
}


def get_eic_config_file_path():
    if os.environ.get(REMOTE_EIC_YAML_ENV_VAR) is not None:
        logger.info(f"eic init with env var {REMOTE_EIC_YAML_ENV_VAR}")
        config_file = os.environ.get(REMOTE_EIC_YAML_ENV_VAR)
    else:
        config_file = "/sgl-workspace/config/remote-eic.yaml"
        logger.info(f"eic init with default config, config_file {config_file}")
    return config_file


class MemoryStateInt(IntEnum):
    IDLE = 0
    RESERVED = 1
    PROTECTED = 2
    SYNCED = 3
    BACKUP = 4


class FlexibleKVCacheMemoryPool:
    def __init__(self, conn, kvcache_shape, kvcache_dtype, device):
        self.connection = conn
        self.kvcache_shape = kvcache_shape
        self.kvcache_dtype = kvcache_dtype

        self.kv_cache_numel = 1
        for i in kvcache_shape:
            self.kv_cache_numel *= i

        device = str(device)
        if device.startswith("cpu") and G_EnableGPUNicAffinity:
            gpu_id = torch.cuda.current_device()
            self.device = CPUNicAffinity.get("cuda:" + str(gpu_id), "cpu")
            # current memory pool size is 5 times of CPU buffer size
            mempool_size = G_GDRBounceTensorCount * 5
        else:
            self.device = device
            mempool_size = G_GDRBounceTensorCount

        self.free_data_addr = set()
        self.data_ptr_to_index = dict()

        if self.device.startswith("cpu"):
            self.kvcache_mempool = torch.zeros(
                (mempool_size,) + kvcache_shape,
                dtype=kvcache_dtype,
                device=self.device,
                pin_memory=True,
            )
        else:
            self.kvcache_mempool = torch.zeros(
                (mempool_size,) + kvcache_shape, dtype=kvcache_dtype, device=self.device
            )
        for i in range(mempool_size):
            self.free_data_addr.add(i)
            self.data_ptr_to_index[self.kvcache_mempool[i].data_ptr()] = i

        meminfo = eic.MemoryInfo()
        meminfo.type = eic.MemoryType.MEMORY_CUDA
        meminfo.cuda_id = torch.cuda.current_device()
        vals = eic.IOBuffers()
        vals.append(
            self.kvcache_mempool.data_ptr(),
            self.kvcache_mempool.numel() * self.kvcache_mempool.element_size(),
            True,
        )
        if not self.connection.register_memory(vals, meminfo):
            # A failed registration is otherwise silent: values are still appended with
            # register_memory=True, so QueryMR misses and byterpc's ReplaceIOBlockRef
            # falls back to a CPU memcpy into a bounce buffer, an extra copy per block
            # on the IO thread. Fail at startup instead.
            raise RuntimeError(
                f"eic register_memory failed for a {self.device} pool of "
                f"{self.kvcache_mempool.numel() * self.kvcache_mempool.element_size()} bytes"
            )
        logger.info(
            f"allocate memory pool, size {self.kvcache_mempool.numel() * self.kvcache_mempool.element_size()}, device {self.device}"
        )

    def try_allocate_kv_cache(self, shape, dtype, count=1):
        if len(self.free_data_addr) < count:
            return None, None

        numel = 1
        for i in shape:
            numel *= i
        if numel != self.kv_cache_numel or dtype != self.kvcache_dtype:
            logger.error(
                f"allocate from mempool failed, self.kvcache_shape {self.kvcache_shape}, dtype {self.kvcache_dtype}, require shape {shape}, dtype {dtype}"
            )
            return None, None

        ret = []
        indices = []
        for _ in range(count):
            free_index = self.free_data_addr.pop()
            ret.append(self.kvcache_mempool[free_index])
            indices.append(free_index)
        return ret, indices

    def free_to_mempool(self, data_ptr):
        if data_ptr not in self.data_ptr_to_index:
            logger.error(
                f"free_to_mempool failed, data_ptr {data_ptr} not in allocated_data_addr"
            )
            return
        self.free_data_addr.add(self.data_ptr_to_index[data_ptr])

    def check_data_ptr_allocated(self, data_ptr):
        return data_ptr in self.data_ptr_to_index

    def left_count(self):
        return len(self.free_data_addr)


class EICKVClient:
    """
    The remote url should start with "eic://" and only have one host-port pair
    """

    def __init__(self, kv_cache_dtype, kv_cache_shape, device_pool, device="cpu"):
        if eic is None:
            raise ImportError("eic module is not found, please install eic package")
        global G_EnableKVSetGPUDirect, G_EnableKVGetGPUDirect, G_EnableAsyncKVSet
        global GPUNicAffinity, CPUNicAffinity, G_EnableGPUNicAffinity, G_KVSetTTLOption
        global G_GDRBounceBufferSize, G_GDRBounceTensorCount

        config_file = get_eic_config_file_path()
        if os.path.exists(config_file) is False:
            logger.error(f"config file {config_file} not exists")
            exit(1)

        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        if remote_url is None:
            raise RuntimeError("remote_url is None")

        endpoint = remote_url[len("eic://") :]

        logger.info(f"eic remote_url:" + remote_url + " endpoint: " + endpoint)

        eic_instance_id = config.get("eic_instance_id", "")
        logger.info(f"eic instance_id: {eic_instance_id}")

        eic_thread_num = config.get("eic_thread_num", 1)
        logger.info(f"eic thread_num: {eic_thread_num}")

        eic_log_dir = config.get("eic_log_dir", "/tmp/eic-client-log")
        logger.info(f"eic log_dir: {eic_log_dir}")

        eic_log_level = config.get("eic_log_level", 2)
        logger.info(f"eic log_level: {eic_log_level}")

        eic_trans_type = config.get("eic_trans_type", 3)
        logger.info(f"eic trans_type: {eic_trans_type}")

        eic_flag_file = config.get("eic_flag_file", None)
        logger.info(f"eic flag_file: {eic_flag_file}")

        # The write pool stays on the host regardless of enable_kvset_gpu_direct:
        # pointing it at CUDA kills the server. The client segfaults on its own IO
        # thread (be::Engine::Transmit) and the other ranks then fail the write-path
        # all_reduce; measured in the 2026-09-12 snapshot, "Write-path config A/B".
        # eic 1.5.2 gives the remote write path no CUDA handling to fall back on:
        # Client::register_memory discards the MemoryInfo it is handed (cuda_id
        # included) and registers every region as byterpc HolderType::UserNormal,
        # byterpc v1.1 having no CUDA holder type, and nothing under
        # src/client/io/remote_engine reads memory_info at all -- the client's CUDA
        # staging (CudaDeepCopyIOBuf) exists only on the local cache path. Reads take
        # a different path, one-sided RDMA into the device buffer, which is why
        # enable_kvget_gpu_direct is safe and this is not.
        if config.get("enable_kvset_gpu_direct", False):
            logger.warning(
                "eic enable_kvset_gpu_direct is ignored: the client cannot register "
                "CUDA memory for the remote write path and segfaults in its IO thread"
            )
        G_EnableKVSetGPUDirect = False
        logger.info(f"eic enable_kvset_gpu_direct: {G_EnableKVSetGPUDirect}")

        G_EnableKVGetGPUDirect = (
            config.get("enable_kvget_gpu_direct", True) and torch.cuda.is_available()
        )
        logger.info(f"eic enable_kvget_gpu_direct: {G_EnableKVGetGPUDirect}")

        # rdma write
        enable_kv_set_direct = config.get("enable_kvset_direct", True)
        logger.info(f"eic enable_kv_set_direct: {enable_kv_set_direct}")
        self.enable_kv_set_direct = enable_kv_set_direct

        # gpu nic affinity
        G_EnableGPUNicAffinity = config.get("enable_gpu_nic_affinity", False)
        logger.info(f"eic enable_gpu_nic_affinity: {G_EnableGPUNicAffinity}")
        self.enable_gpu_nic_affinity = G_EnableGPUNicAffinity

        G_KVSetTTLOption = config.get("kv_set_ttl_option", -1)
        logger.info(f"eic kv set ttl: {G_KVSetTTLOption}")
        self.kv_set_ttl_option = G_KVSetTTLOption

        if G_EnableGPUNicAffinity:
            if "gpu_nic_affinity_config" in config:
                GPUNicAffinity = json.loads(config["gpu_nic_affinity_config"])
            if "cpu_nic_affinity_config" in config:
                CPUNicAffinity = json.loads(config["cpu_nic_affinity_config"])
            logger.info(f"eic gpu nic affinity {GPUNicAffinity}")
            logger.info(f"eic cpu nic affinity {CPUNicAffinity}")

        G_EnableAsyncKVSet = config.get("enable_async_kvset", False)
        logger.info(f"eic enable_async_kvset: {G_EnableAsyncKVSet}")

        eic_namespace = config.get("eic_namespace", "")
        logger.info(f"eic namespace: {eic_namespace}")
        self.eic_namespace = eic_namespace

        self.gdr_bounce_buffer_size = config.get(
            "gdr_bounce_buffer_size", G_GDRBounceBufferSize
        )
        G_GDRBounceBufferSize = self.gdr_bounce_buffer_size
        logger.info(f"eic gdr_bounce_buffer_size: {self.gdr_bounce_buffer_size}")

        kv_cache_numel = 1
        for i in kv_cache_shape:
            kv_cache_numel *= i
        G_GDRBounceTensorCount = self.gdr_bounce_buffer_size // (
            kv_cache_numel * kv_cache_dtype.itemsize
        )
        G_GDRBounceTensorCount = max(1, G_GDRBounceTensorCount)
        eic_max_batch_size = config.get("eic_max_batch_size", None)
        if eic_max_batch_size is not None:
            G_GDRBounceTensorCount = min(
                G_GDRBounceTensorCount, max(1, int(eic_max_batch_size))
            )
        logger.info(
            f"eic gdr_bounce_tensor_count: {G_GDRBounceTensorCount}, kvcache_size {kv_cache_numel * kv_cache_dtype.itemsize}"
        )

        # other configurations
        self.eic_check_bs = config.get("eic_check_bs", 2048)
        logger.debug(f"eic check bs: {self.eic_check_bs}")
        self.eic_direct_backup = config.get("eic_direct_backup", False)
        logger.debug(f"eic direct backup: {self.eic_direct_backup}")
        self.eic_direct_writeback = (
            config.get("eic_direct_writeback", False) and G_EnableKVGetGPUDirect
        )
        logger.debug(f"eic direct writeback: {self.eic_direct_writeback}")

        if not os.path.exists(eic_log_dir) and not os.path.isdir(eic_log_dir):
            os.makedirs(eic_log_dir, exist_ok=True)

        self.connection = eic.Client()
        init_option = eic.InitOption()
        init_option.log_dir = eic_log_dir
        init_option.log_level = eic.LogLevel(eic_log_level)
        init_option.transport_type = eic.TransportType(eic_trans_type)
        if eic_flag_file is not None:
            init_option.flag_file = eic_flag_file

        if G_EnableGPUNicAffinity:
            gpu_id = torch.cuda.current_device()
            init_option.multi_net_local_interface_names = GPUNicAffinity.get(
                "cuda:" + str(gpu_id), "eth1"
            )
            logger.info(
                f"gpu {gpu_id} set gpu nic affinity to {init_option.multi_net_local_interface_names}"
            )

        ret = self.connection.init(eic_instance_id, endpoint, init_option)
        if ret != 0:
            logger.error(f"fail to init eic client, ret: {ret}")
            exit(1)

        self.device = device
        self.trans_type = eic.TransportType(eic_trans_type)
        self.kv_cache_shape = kv_cache_shape
        self.kv_cache_dtype = kv_cache_dtype
        self.device_pool = device_pool

        # use for kv get
        self.kv_cache_read_mem_pool = FlexibleKVCacheMemoryPool(
            self.connection,
            self.kv_cache_shape,
            self.kv_cache_dtype,
            self.device if G_EnableKVGetGPUDirect else "cpu",
        )

        if G_EnableAsyncKVSet:
            logger.info("enable async kv set")
            self.kv_cache_write_mem_pool = FlexibleKVCacheMemoryPool(
                self.connection,
                self.kv_cache_shape,
                self.kv_cache_dtype,
                "cpu",
            )
        else:
            logger.info("enable sync kv set")
            self.kv_cache_write_mem_pool = FlexibleKVCacheMemoryPool(
                self.connection,
                self.kv_cache_shape,
                self.kv_cache_dtype,
                self.device if G_EnableKVSetGPUDirect else "cpu",
            )

        self.write_queue = queue.Queue()
        self._write_drop_ct = 0
        self._write_thread_num = 1
        self._write_thread_pool = [
            threading.Thread(target=self._write_thread, args=())
            for _ in range(self._write_thread_num)
        ]
        for thread in self._write_thread_pool:
            thread.start()

        self._warm_up()

    def _warm_up(self):
        logger.info("begin warm up eic client")
        start_time = time.perf_counter()
        num_warmup = 1024
        preheat_keys = ["warmup_key_" + str(i) for i in range(num_warmup)]
        batch_size = 32
        for i in range(0, num_warmup, batch_size):
            keys_vec = eic.StringVector()
            for key in preheat_keys[i : i + batch_size]:
                keys_vec.append(key)
            exist_option = eic.ExistOption()
            status_code, exist_outcome = self.connection.mexist(keys_vec, exist_option)
        logger.info(
            f"finish eic client warm up, warm up cost {time.perf_counter() - start_time:.2f} seconds"
        )

    def _write_thread(self):
        logger.info(f"start write thread thread_id {threading.get_ident()}")
        while True:
            keys, values, copy_event = self.write_queue.get()
            try:
                if copy_event is not None:
                    copy_event.synchronize()
                self._async_set_impl(keys, values)
            except Exception:
                logger.exception("eic async write failed, dropping %d keys", len(keys))
            finally:
                for value in values:
                    if self.kv_cache_write_mem_pool.check_data_ptr_allocated(
                        value.data_ptr()
                    ):
                        self.kv_cache_write_mem_pool.free_to_mempool(value.data_ptr())

    def async_batch_set(
        self,
        keys: List[str],
        obj_inputs: List[torch.Tensor],
        device_indices: torch.Tensor = None,
        copy_func=None,
    ) -> bool:
        logger.debug(f"eic async_batch_set {len(keys)}")
        start_time = time.perf_counter()
        count = len(keys)

        if self.kv_cache_write_mem_pool.left_count() >= count:
            objs, host_indices = self.kv_cache_write_mem_pool.try_allocate_kv_cache(
                self.kv_cache_shape, self.kv_cache_dtype, count
            )
            if objs is None:
                return False

            copy_event = None
            if obj_inputs is not None:
                write_stream = torch.cuda.current_stream()
                with torch.cuda.stream(write_stream):
                    for i in range(len(keys)):
                        objs[i] = objs[i].reshape(obj_inputs[i].shape)
                        objs[i].copy_(obj_inputs[i], non_blocking=True)
                copy_event = torch.cuda.Event()
                copy_event.record(write_stream)
            elif device_indices is not None and copy_func is not None:
                copy_func(device_indices, objs)
            else:
                logger.error(
                    "async set impl input obj_inputs and device_indices are both None"
                )
                return False
            self.write_queue.put((keys, objs, copy_event))

            cost_time = time.perf_counter() - start_time
            if cost_time >= 0.5:
                logger.info(
                    f"async batch set cost { cost_time * 1e3}.2f ms, {len(keys)} keys"
                )
            return True
        else:
            # Write pool exhausted: the consumer thread is behind (EIC backend slow
            # or failing). Drop this batch instead of allocating an unbounded host
            # tensor per call -- under a backend stall the old fallback accumulated
            # 1000+ large mallocs in seconds and OOM-killed the rank. Dropping is a
            # benign L2/L3 cache miss (recomputed later). Rank-safe: assign_page_data
            # returns False without blocking, so every TP rank still reaches the
            # write_operation_shared all_reduce, which reconciles the failed result.
            self._write_drop_ct += 1
            if self._write_drop_ct == 1 or self._write_drop_ct % 128 == 0:
                logger.warning(
                    "eic async write pool exhausted, dropping %d keys "
                    "(total dropped batches: %d)",
                    len(keys),
                    self._write_drop_ct,
                )
            return False

    def exists(self, key: str) -> bool:
        logger.debug(f"eic exists {key}")
        keys = eic.StringVector()
        keys.append(key)
        exist_option = eic.ExistOption()
        exist_option.ns = self.eic_namespace

        status_code, exist_outcome = self.connection.mexist(keys, exist_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.debug(f"eic exists {key} failed, status_code {status_code}")

        err_code = exist_outcome.status_codes[0]
        success = err_code == eic.StatusCode.SUCCESS
        if success:
            logger.debug(f"eic exists {key} success")
        else:
            logger.debug(f"eic exists {key} failed, err_code {err_code}")
        return success

    def exists_batch(self, keys: List[str]) -> List[bool]:
        logger.debug(f"eic exists {len(keys)}")
        keys_vec = eic.StringVector()
        for key in keys:
            keys_vec.append(key)
        exist_option = eic.ExistOption()
        exist_option.ns = self.eic_namespace

        status_code, exist_outcome = self.connection.mexist(keys_vec, exist_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic exists {len(keys)} failed, status_code {status_code}")
            return [False] * len(keys)
        res = []
        for err_code in exist_outcome.status_codes:
            res.append(err_code == eic.StatusCode.SUCCESS)
        return res

    def allocate_eic_read_buffer(self, count):
        registered = True
        tensors, indices = self.kv_cache_read_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        host_memory_pool = self.kv_cache_read_mem_pool.kvcache_mempool
        if tensors is None:
            logger.error("can not allocate tensor from pool")
            registered = False
            kvcache_tensor = torch.empty(
                (count,) + self.kv_cache_shape, dtype=self.kv_cache_dtype, device="cpu"
            )
            tensors = [kvcache_tensor[i] for i in range(count)]
            host_memory_pool = kvcache_tensor
            indices = range(count)
        return tensors, host_memory_pool, indices, registered

    def _refetch_failed(self, keys, objs, registered, get_option, outcome):
        # Partial mget failures are per-key RPC timeouts on a loaded backend, and
        # the load keeps only the prefix before the first failed key. Re-getting
        # just the failed keys once, into their own buffers, saves the rest from
        # recompute. A missing status counts as failed. ponytail: one retry, only
        # when at most half failed; per-key backoff if the backend flaps.
        codes = list(outcome.status_codes)[: len(keys)]
        codes += [None] * (len(keys) - len(codes))
        failed = [i for i, c in enumerate(codes) if c != eic.StatusCode.SUCCESS]
        if 2 * len(failed) > len(keys):
            # Most keys failing means the backend itself is down; a retry would
            # only stall the serial load thread for another timeout round.
            return eic.StatusCode.PARTIAL_FAILED, SimpleNamespace(status_codes=codes)
        retry_keys, retry_vals = eic.StringVector(), eic.IOBuffers()
        for i in failed:
            retry_keys.append(keys[i])
            retry_vals.append(
                objs[i].data_ptr(), objs[i].element_size() * objs[i].numel(), registered
            )
        status, _, retry_outcome = self.connection.mget(
            retry_keys, get_option, retry_vals
        )
        retry_codes = list(retry_outcome.status_codes)
        for j, i in enumerate(failed):
            if status == eic.StatusCode.SUCCESS or (
                status == eic.StatusCode.PARTIAL_FAILED
                and j < len(retry_codes)
                and retry_codes[j] == eic.StatusCode.SUCCESS
            ):
                codes[i] = eic.StatusCode.SUCCESS
        recovered = sum(codes[i] == eic.StatusCode.SUCCESS for i in failed)
        logger.info(
            f"eic mget refetched {len(failed)} failed keys, {recovered} recovered"
        )
        ok = all(c == eic.StatusCode.SUCCESS for c in codes)
        status = eic.StatusCode.SUCCESS if ok else eic.StatusCode.PARTIAL_FAILED
        return status, SimpleNamespace(status_codes=codes)

    def batch_get(
        self, keys: List[str], device_indices: torch.Tensor = None, copy_func=None
    ) -> Tuple[Optional[List[torch.Tensor]], List[bool]]:
        logger.debug(f"eic get {len(keys)}")

        # Get Data: generate data keys and vals
        get_data_start_time = time.perf_counter()
        data_keys = eic.StringVector()
        data_vals = eic.IOBuffers()
        objs = None
        success_mask = [True for _ in range(len(keys))]
        count = len(keys)

        objs, host_memory_pool, host_indices, registered = (
            self.allocate_eic_read_buffer(count)
        )

        for i, key in enumerate(keys):
            data_keys.append(key)
            data_vals.append(
                objs[i].data_ptr(), objs[i].element_size() * objs[i].numel(), registered
            )

        # Get data: recv data buffer tensor
        get_option = eic.GetOption()
        get_option.ns = self.eic_namespace
        status_code, data_vals, get_outcome = self.connection.mget(
            data_keys, get_option, data_vals
        )
        if status_code == eic.StatusCode.PARTIAL_FAILED:
            status_code, get_outcome = self._refetch_failed(
                keys, objs, registered, get_option, get_outcome
            )

        result = []
        device_copy = False
        # copy before free
        if device_indices is None or copy_func is None:
            if registered:
                for obj in objs:
                    result.append(obj.clone())
            else:
                result = objs
        else:
            device_copy = True

        fail_count = 0
        if status_code != eic.StatusCode.SUCCESS:
            if status_code == eic.StatusCode.PARTIAL_FAILED:
                for i, err_code in enumerate(get_outcome.status_codes):
                    success = err_code == eic.StatusCode.SUCCESS
                    if success:
                        logger.debug(f"eic get data {keys[i]} success")
                    else:
                        logger.debug(
                            f"eic get data {keys[i]} failed, err_code {err_code}"
                        )
                        success_mask[i] = False
                        fail_count += 1
            else:
                logger.error(
                    f"eic mget {len(keys)} keys failed, status_code {status_code}"
                )
                result = None
                success_mask = [False for _ in range(len(keys))]
                fail_count = len(keys)

        if fail_count != 0:
            # Note that fail count only means how many keys failed in this batch get, not really failed in prefix cache
            logger.warning(
                f"eic mget {len(keys)} keys failed, fail count {fail_count}, success count {count - fail_count}"
            )
        if device_copy:
            suc_count = 0
            for mask in success_mask:
                if mask:
                    suc_count += 1
                else:
                    break
            if suc_count > 0:
                copy_func(device_indices, host_memory_pool, host_indices[:suc_count])

        if registered:
            for item in objs:
                self.kv_cache_read_mem_pool.free_to_mempool(item.data_ptr())

        get_data_end_time = time.perf_counter()
        get_data_execution_time = (get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"eic get {count} keys data cost %.2f us", get_data_execution_time)
        return result, success_mask

    def allocate_eic_write_buffer(self, count):
        registered = True
        objs, _ = self.kv_cache_write_mem_pool.try_allocate_kv_cache(
            self.kv_cache_shape, self.kv_cache_dtype, count
        )
        if objs is None:
            logger.error("can not allocate tensor from pool")
            registered = False
            kvcache_tensor = torch.empty(
                (count,) + self.kv_cache_shape, dtype=self.kv_cache_dtype, device="cpu"
            )
            objs = [kvcache_tensor[i] for i in range(count)]
        return objs, registered

    def set(
        self,
        keys: List[str],
        obj_inputs: List[torch.Tensor],
        device_indices: torch.Tensor = None,
        copy_func=None,
    ) -> bool:
        logger.debug(f"eic set {len(keys)}")
        return self._sync_set_impl(keys, obj_inputs, device_indices, copy_func)

    def _sync_set_impl(
        self,
        keys: List[str],
        obj_inputs: List[torch.Tensor],
        device_indices: torch.Tensor = None,
        copy_func=None,
    ) -> bool:
        logger.debug(f"eic set {len(keys)} keys")
        keys_vec = eic.StringVector()
        vals_vec = eic.IOBuffers()
        count = len(keys)

        objs, registered = self.allocate_eic_write_buffer(count)
        values_data_ptrs = []

        if obj_inputs is not None:
            write_stream = torch.cuda.current_stream()
            with torch.cuda.stream(write_stream):
                for i, key in enumerate(keys):
                    temp = objs[i].reshape(obj_inputs[i].shape).contiguous()
                    temp.copy_(obj_inputs[i], non_blocking=True)

                    if temp.data_ptr() != objs[i].data_ptr():
                        registered = False
                        temp = temp.cpu()
                    values_data_ptrs.append(
                        (
                            temp.data_ptr(),
                            temp.element_size() * temp.numel(),
                            registered,
                        )
                    )
            write_stream.synchronize()
        elif device_indices is not None and copy_func is not None:
            copy_func(device_indices, objs)
            for i, key in enumerate(keys):
                values_data_ptrs.append(
                    (
                        objs[i].data_ptr(),
                        objs[i].element_size() * objs[i].numel(),
                        registered,
                    )
                )
        else:
            logger.error("set impl input obj_inputs and device_indices are both None")
            return False

        for i, key in enumerate(keys):
            keys_vec.append(key)
            data_ptr, data_size, registered = values_data_ptrs[i]
            vals_vec.append(
                data_ptr, data_size, registered and self.enable_kv_set_direct
            )

        # set options
        set_option = eic.SetOption()
        set_option.ns = self.eic_namespace
        set_option.ttl_second = self.kv_set_ttl_option
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)

        # Free slots before any early exit: the write mempool never refills, so an
        # mset-error bail leaks them permanently. Test each item rather than the
        # `registered` flag -- the loops above reassign it per key, so it carries
        # only the last key's state and a single reshape fallback would strand the
        # whole batch.
        for item in objs:
            if self.kv_cache_write_mem_pool.check_data_ptr_allocated(item.data_ptr()):
                self.kv_cache_write_mem_pool.free_to_mempool(item.data_ptr())

        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {len(keys)} failed, status_code {status_code}")
            return False
        else:
            logger.debug(f"eic mset {len(keys)} success")

        failed = [
            (i, err_code)
            for i, err_code in enumerate(set_outcome.status_codes)
            if err_code != eic.StatusCode.SUCCESS
        ]
        if failed:
            logger.error(
                "set data key batch failed, total=%s failed=%s first_failed=%s",
                len(keys),
                len(failed),
                failed[0],
            )
            return False

        logger.debug(f"set data key {len(keys)} success")
        return True

    def _async_set_impl(self, keys: List[str], obj_inputs: List[torch.Tensor]) -> None:
        logger.debug(f"eic set {len(keys)} keys")
        keys_vec = eic.StringVector()
        vals_vec = eic.IOBuffers()

        # set data key & value
        for key, value in zip(keys, obj_inputs):
            obj = value
            # set data key & value
            keys_vec.append(key)
            registered = self.kv_cache_write_mem_pool.check_data_ptr_allocated(
                obj.data_ptr()
            )
            vals_vec.append(
                obj.data_ptr(),
                obj.element_size() * obj.numel(),
                registered,
            )

        # set options
        set_option = eic.SetOption()
        set_option.ns = self.eic_namespace
        set_option.ttl_second = self.kv_set_ttl_option
        status_code, set_outcome = self.connection.mset(keys_vec, vals_vec, set_option)

        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {len(keys)} failed, status_code {status_code}")
            return False
        else:
            logger.debug(f"eic mset {len(keys)} success")

        failed = [
            (i, err_code)
            for i, err_code in enumerate(set_outcome.status_codes)
            if err_code != eic.StatusCode.SUCCESS
        ]
        if failed:
            logger.error(
                "async set data key batch failed, total=%s failed=%s first_failed=%s",
                len(keys),
                len(failed),
                failed[0],
            )
            return False
        return status_code == eic.StatusCode.SUCCESS


class EICBaseTokenToKVPoolHost:
    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float = 4.0,
        host_size: int = 10,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        self.device_pool = device_pool
        self.host_to_device_ratio = host_to_device_ratio
        self.device = device
        self.dtype = device_pool.store_dtype
        self.page_size = page_size
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.size = self.size - (self.size % self.page_size)
        # The metrics reporter reads logical_size off whatever host pool the tree
        # cache exposes. EIC pools are flat -- no anchor entry to proxy -- so the
        # logical size is just size.
        self.logical_size = self.size

        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        self.free_slots = torch.arange(self.size, dtype=torch.int32)
        self.can_use_mem_size = self.size

        # A lock for synchronized() operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.debug = logger.isEnabledFor(logging.DEBUG)

        self.rank = rank
        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        self.host_ip = self._get_host_ip()
        self.split_dim = 2
        self.split_size = self.page_size
        self.extra_info = extra_info
        self.deploy_key = self._get_deploy_info()
        self.device_backup_func = self.device_backup
        self.device_writeback_func = self.device_writeback

    def _encode_key_exclusive(self, indices):
        return [
            f"{self.host_ip}_{self.rank}_{index}"
            for index in indices.to("cpu").tolist()
        ]

    def _get_host_ip(self):
        import os
        import socket

        # gethostbyname can resolve to 127.0.0.1 on some hosts, colliding keys
        # across nodes; prefer the pod-provided MY_HOST_IP.
        return os.environ.get("MY_HOST_IP") or socket.gethostbyname(
            socket.gethostname()
        )

    def _get_deploy_info(self):
        model_path = self.extra_info.get("model_path", "fake_model_path")
        attn_tp_size = self.attn_tp_size
        attn_tp_rank = self.attn_tp_rank
        page_size = self.page_size
        dtype = self.dtype
        framework = self.extra_info.get("framework", "sglang")
        pp_size = self.extra_info.get("pp_size", 1)
        if pp_size > 1:
            pp_rank = self.extra_info.get("pp_rank", 0)
            deploy_key = f"{model_path}_{pp_size}_{pp_rank}_{attn_tp_size}_{attn_tp_rank}_{page_size}_{dtype}@{framework}"
        else:
            deploy_key = f"{model_path}_{attn_tp_size}_{attn_tp_rank}_{page_size}_{dtype}@{framework}"
        return deploy_key

    def _encode_key_shared(self, content_hashs):
        return [f"{content_hash}@{self.deploy_key}" for content_hash in content_hashs]

    def batch_exist_page(self, content_hashes):
        """
        for multi prompt detect prefix key
        """
        batch_size = self.eic_client.eic_check_bs
        keys = self._encode_key_shared(content_hashes)

        exist_result = []
        for i in range(0, len(keys), batch_size):
            batch_ret = self.eic_client.exists_batch(keys[i : i + batch_size])
            exist_result.extend(batch_ret)
        return exist_result

    def get_flat_data_direct(self, keys, device_indices=None):
        bs = G_GDRBounceTensorCount // 2
        masks = []

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            indices = device_indices[i : i + bs]
            _, success_mask = self.eic_client.batch_get(
                key, indices, self.device_writeback_func
            )
            masks.extend(success_mask)
            if not all(success_mask):
                break
        return masks

    def get_flat_data_common(self, keys, device_indices=None):
        bs = G_GDRBounceTensorCount // 2
        ret = []
        indexer_ret = []
        masks = []
        suc_count = 0

        for i in range(0, len(keys), bs):
            key = keys[i : i + bs]
            objs, success_mask = self.eic_client.batch_get(key)
            if objs is None:
                logger.debug(f"get_page_data keys {key} failed, eic_client return none")
                break
            # todo(zhongwei.ren): preallocate recv tensors
            ret.extend(objs)
            masks.extend(success_mask)

            if not all(success_mask):
                break

        for mask in masks:
            if mask:
                suc_count += 1
            else:
                break
        if suc_count > 0:
            flat_data = torch.cat(ret[:suc_count], dim=self.split_dim)
            self.device_pool.transfer(
                device_indices[: suc_count * self.page_size], flat_data
            )
        return masks

    def get_flat_data(self, indices) -> Tuple[Optional[torch.Tensor], List[bool]]:
        logger.debug(f"get_flat_data indices {indices}")
        keys = self._encode_key_exclusive(indices)
        if self.eic_client.eic_direct_writeback:
            return self.get_flat_data_direct(keys, indices)
        else:
            return self.get_flat_data_common(keys, indices)

    def assign_flat_data_common(self, keys, flat_data, device_indices=None) -> bool:
        if flat_data is None:
            flat_data = self.device_pool.get_flat_data(device_indices)
        start_time = time.perf_counter()
        flat_data = flat_data.contiguous()
        values = torch.split(flat_data, 1, dim=self.split_dim)

        bs = G_GDRBounceTensorCount
        split_time = time.perf_counter()
        count = len(keys)
        for i in range(0, count, bs):
            key = keys[i : i + bs]
            value = values[i : i + bs]
            if G_EnableAsyncKVSet:
                ret = self.eic_client.async_batch_set(key, value)
            else:
                ret = self.eic_client.set(key, value)
            if not ret:
                logger.error(
                    f"assign_flat_data keys {key} failed, eic_client return none"
                )
                return False
        cost_time = time.perf_counter() - split_time
        if cost_time > 1:
            logger.warning(
                f"finish assign flat data, total keys {len(keys)}, split time {split_time - start_time}, transfer time {cost_time}"
            )
        return True

    def assign_flat_data_direct(self, keys, device_indices=None) -> bool:
        if device_indices is None:
            logger.error("assign_flat_data_direct device_indices is None")
            return False

        bs = G_GDRBounceTensorCount
        start_time = time.perf_counter()
        count = len(keys)
        for i in range(0, count, bs):
            key = keys[i : i + bs]
            indices = device_indices[i : i + bs]
            if G_EnableAsyncKVSet:
                ret = self.eic_client.async_batch_set(
                    key, None, indices, copy_func=self.device_backup_func
                )
            else:
                ret = self.eic_client.set(
                    key, None, indices, copy_func=self.device_backup_func
                )
            if not ret:
                logger.error(
                    f"assign_flat_data keys {key} failed, eic_client return none"
                )
                return False
        cost_time = time.perf_counter() - start_time
        if cost_time > 1:
            logger.warning(
                f"finish assign flat data, total keys {len(keys)}, transfer time {cost_time}"
            )
        return True

    def assign_flat_data(self, indices, flat_data, device_indices=None) -> bool:
        logger.debug(f"assign_flat_data indices {indices}")
        keys = self._encode_key_exclusive(indices)
        if self.eic_client.eic_direct_backup and flat_data is None:
            return self.assign_flat_data_direct(keys, device_indices)
        else:
            return self.assign_flat_data_common(keys, flat_data, device_indices)

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_page_data_direct(self, keys, device_indices=None):
        bs = G_GDRBounceTensorCount
        masks = []
        count = len(keys)
        for i in range(0, count, bs):
            key = keys[i : i + bs]
            indices = device_indices[i : i + bs * self.page_size]
            _, success_mask = self.eic_client.batch_get(
                key, indices, self.device_writeback_func
            )
            masks.extend(success_mask)
            if not all(success_mask):
                break
        return masks

    def get_page_data_common(self, keys, device_indices=None):
        count = len(keys)
        bs = G_GDRBounceTensorCount // 2
        ret = []
        indexer_ret = []
        masks = []
        suc_count = 0

        for i in range(0, count, bs):
            key = keys[i : i + bs]
            objs, success_mask = self.eic_client.batch_get(key)
            if objs is None:
                logger.debug(f"get_page_data keys {key} failed, eic_client return none")
                break
            # todo(zhongwei.ren): preallocate recv tensors
            ret.extend(objs)
            masks.extend(success_mask)

            if not all(success_mask):
                break

        for mask in masks:
            if mask:
                suc_count += 1
            else:
                break

        if suc_count > 0:
            flat_data = torch.cat(ret[:suc_count], dim=self.split_dim)
            self.device_pool.transfer(
                device_indices[: suc_count * self.page_size], flat_data
            )
        return masks

    def get_page_data(self, content_hashs, device_indices=None):
        logger.debug(f"get_page_data content_hashs {content_hashs}")
        keys = self._encode_key_shared(content_hashs)
        if self.eic_client.eic_direct_writeback:
            return self.get_page_data_direct(keys, device_indices)
        else:
            return self.get_page_data_common(keys, device_indices)

    def assign_page_data_common(self, keys, flat_data, device_indices=None) -> bool:
        if flat_data is None:
            flat_data = self.device_pool.get_flat_data(device_indices)
        start_time = time.perf_counter()
        flat_data = flat_data.contiguous()
        values = torch.split(flat_data, self.split_size, dim=self.split_dim)

        bs = G_GDRBounceTensorCount
        split_time = time.perf_counter()
        count = len(keys)
        for i in range(0, count, bs):
            key = keys[i : i + bs]
            value = values[i : i + bs]
            if G_EnableAsyncKVSet:
                ret = self.eic_client.async_batch_set(key, value)
            else:
                ret = self.eic_client.set(key, value)
            if not ret:
                logger.error(
                    f"assign_page_data keys {key} failed, eic_client return none"
                )
                return False

        cost_time = time.perf_counter() - split_time
        if cost_time > 1:
            logger.warning(
                f"finish assign page data, total keys {len(keys)}, split time {split_time - start_time}, transfer time {cost_time}"
            )
        return True

    def assign_page_data_direct(self, keys, device_indices=None) -> bool:
        if device_indices is None:
            logger.error("assign_page_data_direct device_indices is None")
            return False
        start_time = time.perf_counter()
        bs = G_GDRBounceTensorCount
        count = len(keys)
        for i in range(0, count, bs):
            key = keys[i : i + bs]
            indices = device_indices[i : i + bs * self.page_size]
            if G_EnableAsyncKVSet:
                ret = self.eic_client.async_batch_set(
                    key, None, indices, copy_func=self.device_backup_func
                )
            else:
                ret = self.eic_client.set(
                    key, None, indices, copy_func=self.device_backup_func
                )
            if not ret:
                logger.error(
                    f"assign_page_data keys {key} failed, eic_client return none"
                )
                return False

        cost_time = time.perf_counter() - start_time
        if cost_time > 1:
            logger.warning(
                f"finish assign page data, total keys {len(keys)}, transfer time {cost_time}"
            )
        return True

    def assign_page_data(self, content_hashes, flat_data, device_indices=None):
        logger.debug(f"assign_page_data hashes {content_hashes}")
        if len(content_hashes) == 0:
            return True
        keys = self._encode_key_shared(content_hashes)
        if self.eic_client.eic_direct_backup and flat_data is None:
            return self.assign_page_data_direct(keys, device_indices)
        else:
            return self.assign_page_data_common(keys, flat_data, device_indices)

    def device_backup(
        self, device_indices: torch.Tensor, dst_tensors: List[torch.Tensor]
    ) -> None:
        raise NotImplementedError

    def device_writeback(
        self,
        device_indices: torch.Tensor,
        src_tensors: List[torch.Tensor],
        masks: List[bool],
    ) -> None:
        raise NotImplementedError

    @synchronized
    def clear(self):
        self.mem_state.fill_(0)
        self.can_use_mem_size = self.size
        self.free_slots = torch.arange(self.size, dtype=torch.int32)

    @synchronized
    def get_state(self, indices: torch.Tensor) -> MemoryStateInt:
        assert len(indices) > 0, "The indices should not be empty"
        states = self.mem_state[indices]
        assert (
            states == states[0]
        ).all(), "The memory slots should have the same state {}".format(states)
        return MemoryStateInt(states[0].item())

    @synchronized
    def alloc(self, need_size: int) -> torch.Tensor:
        if need_size > self.can_use_mem_size:
            return None

        # todo: de-fragementation
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        self.mem_state[select_index] = MemoryStateInt.RESERVED
        self.can_use_mem_size -= need_size

        return select_index

    @synchronized
    def is_reserved(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.RESERVED

    @synchronized
    def is_protected(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.PROTECTED

    @synchronized
    def is_synced(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.SYNCED

    @synchronized
    def is_backup(self, indices: torch.Tensor) -> bool:
        return self.get_state(indices) == MemoryStateInt.BACKUP

    @synchronized
    def update_backup(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.BACKUP

    @synchronized
    def update_synced(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.SYNCED

    @synchronized
    def protect_write(self, indices: torch.Tensor):
        assert self.is_reserved(indices), (
            f"The host memory slots should be RESERVED before write operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized
    def protect_load(self, indices: torch.Tensor):
        self.mem_state[indices] = MemoryStateInt.PROTECTED

    @synchronized
    def complete_io(self, indices: torch.Tensor):
        assert self.is_protected(indices), (
            f"The host memory slots should be PROTECTED during I/O operations. "
            f"Current state: {self.get_state(indices)}"
        )
        self.mem_state[indices] = MemoryStateInt.SYNCED

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.mem_state[indices] = MemoryStateInt.IDLE
        self.free_slots = torch.concat([self.free_slots, indices])
        self.can_use_mem_size += len(indices)
        return len(indices)


class EICMHATokenToKVPoolHost(EICBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )
        self.head_num = device_pool.head_num
        self.head_dim = device_pool.head_dim
        self.layer_num = device_pool.layer_num
        self.size_per_token = (
            self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2
        )
        self.kvcache_shape = (
            2,
            self.layer_num,
            page_size,
            self.head_num,
            self.head_dim,
        )
        self.kvcache_device = device_pool.device
        self.eic_client = EICKVClient(
            self.dtype, self.kvcache_shape, device_pool, self.kvcache_device
        )

    def device_backup(
        self, device_indices: torch.Tensor, dst_tensors: List[torch.Tensor]
    ) -> None:
        assert (
            len(device_indices) == len(dst_tensors) * self.page_size
        ), "Length of device_indices and dst_tensors must be the same"
        count = len(device_indices)
        cpu_tensor = torch.zeros(
            2,
            self.layer_num,
            count,
            self.head_num,
            self.head_dim,
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        for layer_id in range(self.layer_num):
            cpu_tensor[0][layer_id][:].copy_(
                self.device_pool.k_buffer[layer_id][device_indices], non_blocking=True
            )
            cpu_tensor[1][layer_id][:].copy_(
                self.device_pool.v_buffer[layer_id][device_indices], non_blocking=True
            )
        torch.cuda.current_stream().synchronize()
        for i, dst in enumerate(dst_tensors):
            start = i * self.page_size
            end = start + self.page_size
            dst.copy_(cpu_tensor[:, :, start:end, :, :].contiguous())

    def device_writeback(self, device_indices, src_tensor, src_indices):
        assert (
            len(device_indices) >= len(src_indices) * self.page_size
        ), "Length of device_indices and src_tensors must be the same"
        suc_count = len(src_indices)
        device_indices = device_indices[: suc_count * self.page_size]
        for layer_id in range(self.layer_num):
            self.device_pool.k_buffer[layer_id][device_indices] = src_tensor[
                src_indices, 0, layer_id
            ].flatten(0, 1)
            self.device_pool.v_buffer[layer_id][device_indices] = src_tensor[
                src_indices, 1, layer_id
            ].flatten(0, 1)


class EICMLATokenToKVPoolHost(EICBaseTokenToKVPoolHost):
    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.kv_cache_dim = self.device_pool.kv_cache_dim
        self.layer_num = self.device_pool.layer_num
        self.size_per_token = self.kv_cache_dim * 1 * self.dtype.itemsize
        self.kvcache_shape = (
            self.layer_num,
            page_size,
            1,
            self.kv_cache_dim,
        )
        self.kvcache_device = device_pool.device
        self.eic_client = EICKVClient(
            self.dtype, self.kvcache_shape, device_pool, self.kvcache_device
        )
        self.split_dim = 1

    def _get_deploy_info(self):
        model_path = self.extra_info.get("model_path", "fake_model_path")
        page_size = self.page_size
        dtype = self.dtype
        framework = self.extra_info.get("framework", "sglang")
        pp_size = self.extra_info.get("pp_size", 1)
        if pp_size > 1:
            pp_rank = self.extra_info.get("pp_rank", 0)
            deploy_key = (
                f"{model_path}_{pp_size}_{pp_rank}_{page_size}_{dtype}@{framework}"
            )
        else:
            deploy_key = f"{model_path}_{page_size}_{dtype}@{framework}"
        return deploy_key

    def _filter_kv_cache(
        self,
        keys: List[str],
        obj_inputs: torch.Tensor,
        device_indices: torch.Tensor = None,
    ):
        attn_tp_size = self.attn_tp_size
        attn_tp_rank = self.attn_tp_rank

        keys_len = len(keys)
        mean_len = keys_len // attn_tp_size
        remainder = keys_len % attn_tp_size
        tp_keys_len = mean_len + (1 if attn_tp_rank < remainder else 0)
        start = attn_tp_rank * mean_len + min(attn_tp_rank, remainder)
        end = start + tp_keys_len
        logger.debug(f"start: {start}, end: {end}, tp_keys_len: {tp_keys_len}")
        ret_keys = keys[start:end]
        ret_inputs = None
        ret_device_indices = None
        if obj_inputs is not None:
            ret_inputs = obj_inputs.narrow(
                dim=self.split_dim,
                start=start * self.page_size,
                length=tp_keys_len * self.page_size,
            )
        if device_indices is not None:
            ret_device_indices = device_indices.narrow(
                dim=0,
                start=start * self.page_size,
                length=tp_keys_len * self.page_size,
            )

        return ret_keys, ret_inputs, ret_device_indices

    def assign_page_data(self, content_hashes, flat_data, device_indices=None):
        content_hashes, flat_data, device_indices = self._filter_kv_cache(
            content_hashes, flat_data, device_indices
        )
        return super().assign_page_data(content_hashes, flat_data, device_indices)

    def get_size_per_token(self):
        self.kv_cache_dim = self.device_pool.kv_cache_dim
        self.dtype = self.device_pool.store_dtype
        self.layer_num = self.device_pool.layer_num

        return self.kv_cache_dim * 1 * self.dtype.itemsize * self.layer_num

    def device_backup(
        self, device_indices: torch.Tensor, dst_tensors: List[torch.Tensor]
    ) -> None:
        assert (
            len(device_indices) == len(dst_tensors) * self.page_size
        ), "Length of device_indices and dst_tensors must be the same"
        count = len(device_indices)
        cpu_tensor = torch.empty(
            self.layer_num,
            count,
            1,
            self.kv_cache_dim,
            dtype=self.dtype,
            device="cpu",
            pin_memory=True,
        )
        for layer_id in range(self.layer_num):
            cpu_tensor[layer_id].copy_(
                self.device_pool.kv_buffer[layer_id][device_indices], non_blocking=True
            )
        torch.cuda.current_stream().synchronize()
        for i, dst in enumerate(dst_tensors):
            start = i * self.page_size
            end = start + self.page_size
            dst.copy_(cpu_tensor[:, start:end, :, :].contiguous())

    def device_writeback(self, device_indices, src_tensor, src_indices):
        assert (
            len(device_indices) >= len(src_indices) * self.page_size
        ), "Length of device_indices and src_tensors must be the same"
        suc_count = len(src_indices)
        device_indices = device_indices[: suc_count * self.page_size]
        for layer_id in range(self.layer_num):
            self.device_pool.kv_buffer[layer_id][device_indices] = src_tensor[
                src_indices, layer_id
            ].flatten(0, 1)


class EICNSATokenToKVPoolHost(EICMLATokenToKVPoolHost):
    def __init__(
        self,
        device_pool: NSATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
    ):
        EICBaseTokenToKVPoolHost.__init__(
            self,
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim

        self.kvcache_shape = (
            self.layer_num,
            1,
            self.final_dim,
        )
        self.kvcache_device = device_pool.device

        # Pass device_pool=None to treat as a standard single-tensor pool in client
        self.eic_client = EICKVClient(
            torch.uint8, self.kvcache_shape, None, self.kvcache_device
        )
        self.split_dim = 1
        self.split_size = 1

    def get_size_per_token(self):
        self.kv_cache_dim = self.device_pool.kv_cache_dim
        self.layer_num = self.device_pool.layer_num
        self.nsa_index_dim = self.device_pool.index_k_with_scale_buffer[0].shape[-1]
        self.nsa_index_dtype = self.device_pool.index_k_with_scale_buffer_dtype

        self.mla_page_bytes = self.kv_cache_dim * self.dtype.itemsize * self.page_size
        self.nsa_page_bytes = self.nsa_index_dim * self.nsa_index_dtype.itemsize
        self.final_dim = self.mla_page_bytes + self.nsa_page_bytes

        self.size_per_token = self.final_dim * self.layer_num / self.page_size
        return self.size_per_token

    def device_backup(
        self, device_indices: torch.Tensor, dst_tensors: List[torch.Tensor]
    ) -> None:
        assert (
            len(device_indices) == len(dst_tensors) * self.page_size
        ), "Length of device_indices and dst_tensors must be the same"
        count = len(device_indices)
        num_pages = count // self.page_size

        # page_indices for NSA (compressed)
        page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )

        # CPU buffer: (layer_num, num_pages, final_dim)
        cpu_tensor = torch.empty(
            self.layer_num,
            num_pages,
            self.final_dim,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )

        for layer_id in range(self.layer_num):
            mla_data = self.device_pool.kv_buffer[layer_id][device_indices]
            mla_bytes = mla_data.view(num_pages, -1).view(torch.uint8)
            cpu_tensor[layer_id, :, : self.mla_page_bytes].copy_(
                mla_bytes, non_blocking=True
            )

            nsa_packed = self.device_pool.index_k_with_scale_buffer[layer_id][
                page_indices
            ]
            cpu_tensor[layer_id, :, self.mla_page_bytes :].copy_(
                nsa_packed, non_blocking=True
            )

        torch.cuda.current_stream().synchronize()

        for i, dst in enumerate(dst_tensors):
            dst.copy_(cpu_tensor[:, i, :].unsqueeze(1), non_blocking=True)

    def device_writeback(self, device_indices, src_tensor, src_indices):
        assert (
            len(device_indices) >= len(src_indices) * self.page_size
        ), "Length of device_indices and src_tensors must be the same"
        suc_count = len(src_indices)

        real_device_indices = device_indices[: suc_count * self.page_size]
        page_indices = (
            real_device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )

        src_data = src_tensor[src_indices]  # (num_pages, layer_num, 1, final_dim)

        for layer_id in range(self.layer_num):
            layer_data = src_data[:, layer_id, 0, :]  # (num_pages, final_dim)
            mla_bytes = layer_data[:, : self.mla_page_bytes]
            nsa_bytes = layer_data[:, self.mla_page_bytes :]
            mla_part = (
                mla_bytes.reshape(-1).view(self.dtype).reshape(-1, 1, self.kv_cache_dim)
            )
            self.device_pool.kv_buffer[layer_id][real_device_indices] = mla_part
            self.device_pool.index_k_with_scale_buffer[layer_id][
                page_indices
            ] = nsa_bytes


class EICDeepSeekV4TokenToKVPoolHost(EICBaseTokenToKVPoolHost):
    """Native EIC shared-page adapter for DeepSeek V4 hybrid KV pools."""

    _KV_SOURCED_POOLS = {
        PoolName.DEEPSEEK_V4_C4,
        PoolName.DEEPSEEK_V4_C4_INDEXER,
        PoolName.DEEPSEEK_V4_C128,
    }
    _SWA_SOURCED_POOLS = {
        PoolName.SWA,
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    }

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float = 4.0,
        host_size: int = 10,
        device: str = "cpu",
        page_size: int = 1,
        rank: int = 0,
        extra_info: Optional[dict] = None,
        *,
        params=None,
        server_args=None,
        load_cache_event=None,
    ):
        if params is None or server_args is None:
            raise ValueError("DeepSeek V4 EIC host requires cache init params.")
        if server_args.disable_eic_shared:
            raise ValueError("DeepSeek V4 EIC cache requires shared-page mode.")

        self._estimated_page_bytes = self._estimate_page_bytes(device_pool)
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            device,
            page_size,
            rank,
            extra_info,
        )

        from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
            build_deepseek_v4_hicache_stack,
        )

        host_pool_group, staging_controller = build_deepseek_v4_hicache_stack(
            params=params,
            server_args=server_args,
            kvcache=device_pool,
            load_cache_event=load_cache_event,
            storage_backend=None,
            # EIC host pages are device-indexed; see _deepseek_v4_num_host_pages.
            device_indexed=True,
        )
        # The assembler creates a temporary HiCache controller that registers a
        # layer-transfer counter on DSv4. Native EIC load completes before the
        # scheduler resumes the request, so keep the pool on its normal path.
        device_pool.register_layer_transfer_counter(None)

        self.host_pool_group = host_pool_group
        self.io_backend = self._resolve_io_backend(server_args)
        self.layer_num = device_pool.layer_num
        self.kvcache_device = device_pool.device
        self.transfer_layer_num = staging_controller.layer_num
        self.page_specs = self._build_page_specs()
        self.page_bytes = sum(spec["numel"] for spec in self.page_specs)
        if self.page_bytes != self._estimated_page_bytes:
            logger.warning(
                "DeepSeek V4 EIC page size estimate mismatch: estimated=%s actual=%s",
                self._estimated_page_bytes,
                self.page_bytes,
            )

        self.eic_chunk_bytes = self._resolve_eic_chunk_bytes()
        self.page_chunk_count = (
            self.page_bytes + self.eic_chunk_bytes - 1
        ) // self.eic_chunk_bytes
        logger.info(
            "DeepSeek V4 EIC page chunking: page_bytes=%s chunk_bytes=%s chunks_per_page=%s",
            self.page_bytes,
            self.eic_chunk_bytes,
            self.page_chunk_count,
        )

        self.dtype = torch.uint8
        self.kvcache_shape = (self.eic_chunk_bytes,)
        self.eic_client = EICKVClient(
            self.dtype, self.kvcache_shape, device_pool, self.kvcache_device
        )

    def _resolve_eic_chunk_bytes(self) -> int:
        chunk_bytes = DEFAULT_DSV4_EIC_CHUNK_BYTES
        config_file = get_eic_config_file_path()
        if os.path.exists(config_file):
            with open(config_file, "r") as fin:
                config = yaml.safe_load(fin) or {}
            chunk_bytes = int(
                config.get(
                    "dsv4_eic_chunk_bytes",
                    config.get("eic_object_chunk_bytes", chunk_bytes),
                )
            )
        if chunk_bytes <= 0:
            chunk_bytes = self.page_bytes
        return max(1, min(chunk_bytes, self.page_bytes))

    def _resolve_io_backend(self, server_args) -> str:
        if server_args.hicache_io_backend:
            return server_args.hicache_io_backend
        if server_args.hicache_mem_layout == "page_first_direct":
            return "direct"
        return "kernel"

    def _estimate_page_bytes(self, device_pool) -> int:
        page_bytes = 0
        if device_pool.swa_kv_pool.kv_buffer:
            page_bytes += len(device_pool.swa_kv_pool.kv_buffer) * int(
                device_pool.swa_kv_pool.bytes_per_page_padded
            )
        if device_pool.c4_kv_pool.kv_buffer:
            page_bytes += len(device_pool.c4_kv_pool.kv_buffer) * int(
                device_pool.c4_kv_pool.bytes_per_page_padded
            )
        c4_indexer_buffer = device_pool.c4_indexer_kv_pool.index_k_with_scale_buffer
        if c4_indexer_buffer:
            page_bytes += len(c4_indexer_buffer) * (
                c4_indexer_buffer[0].shape[1] * c4_indexer_buffer[0].element_size()
            )
        if device_pool.c128_kv_pool.kv_buffer:
            page_bytes += len(device_pool.c128_kv_pool.kv_buffer) * int(
                device_pool.c128_kv_pool.bytes_per_page_padded
            )

        for pools in (
            device_pool.compress_state_pools,
            device_pool.indexer_compress_state_pools,
        ):
            for pool in pools:
                if pool is None or pool.ratio == 128:
                    continue
                state_tensor = pool.kv_score_buffer.kv_score
                page_bytes += int(pool.ring_size * state_tensor[0].nbytes)
        return page_bytes

    def get_size_per_token(self):
        return max(1, self._estimated_page_bytes // max(1, self.page_size))

    def _build_page_specs(self):
        specs = []
        for name, entry in self.host_pool_group.entry_map.items():
            if name == PoolName.KV:
                continue
            host_pool = entry.host_pool
            dummy = host_pool.get_dummy_flat_data_page()
            if dummy.numel() == 0:
                continue
            if name in self._SWA_SOURCED_POOLS:
                source = PoolName.SWA
            elif name in self._KV_SOURCED_POOLS:
                source = PoolName.KV
            else:
                raise ValueError(f"Unsupported DeepSeek V4 EIC pool: {name}")
            specs.append(
                {
                    "name": name,
                    "host_pool": host_pool,
                    "source": source,
                    "numel": dummy.numel(),
                }
            )
        return specs

    def _swa_indices(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.device_pool.translate_loc_from_full_to_swa(full_indices).to(
            torch.int64
        )

    def alloc_device_indices(self, allocator, need_size: int):
        full_allocator = getattr(allocator, "full_attn_allocator", None)
        swa_allocator = getattr(allocator, "swa_attn_allocator", None)
        if full_allocator is None or swa_allocator is None:
            return allocator.alloc(need_size)

        full_indices = full_allocator.alloc(need_size)
        if full_indices is None:
            return None
        # Only the trailing window gets real SWA slots; earlier tokens map to 0.
        swa_window = getattr(self.device_pool, "swa_window_size", need_size)
        num_swa = min(need_size, max(swa_window, self.page_size))
        swa_indices = swa_allocator.alloc(num_swa)
        if swa_indices is None:
            full_allocator.free(full_indices)
            return None
        swa_indices_aliased = torch.zeros(
            len(full_indices), dtype=swa_indices.dtype, device=swa_indices.device
        )
        swa_indices_aliased[-num_swa:] = swa_indices
        allocator.set_full_to_swa_mapping(full_indices, swa_indices_aliased)
        return full_indices

    def _build_pool_transfers(self, full_indices: torch.Tensor):
        full_host_indices = full_indices.detach().cpu().to(torch.int64)
        swa_device_indices = self._swa_indices(full_indices)
        swa_host_indices = swa_device_indices.detach().cpu().to(torch.int64)

        transfers = []
        for spec in self.page_specs:
            if spec["source"] == PoolName.SWA:
                host_indices = swa_host_indices
                device_indices = swa_device_indices
            else:
                host_indices = full_host_indices
                device_indices = full_indices
            transfers.append(
                PoolTransfer(
                    name=spec["name"],
                    host_indices=host_indices,
                    device_indices=device_indices,
                )
            )
        return full_host_indices, swa_host_indices, transfers

    def _pack_page(self, full_page_start: int, swa_page_start: int) -> torch.Tensor:
        pieces = []
        for spec in self.page_specs:
            page_start = (
                swa_page_start if spec["source"] == PoolName.SWA else full_page_start
            )
            pieces.append(
                spec["host_pool"].get_data_page(page_start, flat=True).view(torch.uint8)
            )
        return torch.cat(pieces).contiguous()

    def _unpack_page(
        self, full_page_start: int, swa_page_start: int, page_data: torch.Tensor
    ) -> None:
        offset = 0
        page_data = page_data.view(torch.uint8)
        for spec in self.page_specs:
            next_offset = offset + spec["numel"]
            page_start = (
                swa_page_start if spec["source"] == PoolName.SWA else full_page_start
            )
            spec["host_pool"].set_from_flat_data_page(
                page_start, page_data[offset:next_offset]
            )
            offset = next_offset

    def _encode_page_chunk_keys(self, content_hashes):
        keys = []
        for page_key in self._encode_key_shared(content_hashes):
            for chunk_id in range(self.page_chunk_count):
                keys.append(
                    f"{page_key}#dsv4chunk{chunk_id}of{self.page_chunk_count}b{self.eic_chunk_bytes}"
                )
        return keys

    def batch_exist_page(self, content_hashes):
        if len(content_hashes) == 0:
            return []

        chunk_keys = self._encode_page_chunk_keys(content_hashes)
        batch_size = self.eic_client.eic_check_bs
        chunk_exists = []
        for i in range(0, len(chunk_keys), batch_size):
            chunk_exists.extend(
                self.eic_client.exists_batch(chunk_keys[i : i + batch_size])
            )

        page_exists = []
        for page_id in range(len(content_hashes)):
            start = page_id * self.page_chunk_count
            end = start + self.page_chunk_count
            page_exists.append(all(chunk_exists[start:end]))
        return page_exists

    def device_backup(
        self, device_indices: torch.Tensor, dst_tensors: List[torch.Tensor]
    ) -> None:
        page_count = len(device_indices) // self.page_size
        assert len(device_indices) == page_count * self.page_size
        assert (
            len(dst_tensors) == page_count * self.page_chunk_count
        ), "Length of dst_tensors must equal pages * DSv4 EIC chunks per page"
        full_host_indices, swa_host_indices, transfers = self._build_pool_transfers(
            device_indices
        )
        # Per-pool backup: the new host-pool API removed the group-level
        # backup_from_device_all_layer(..., pool_transfers=) entry point.
        for transfer in transfers:
            entry = self.host_pool_group.get_entry(transfer.name)
            entry.host_pool.backup_from_device_all_layer(
                entry.device_pool,
                transfer.host_indices,
                transfer.device_indices,
                self.io_backend,
            )
        torch.cuda.current_stream().synchronize()

        for page_id in range(page_count):
            token_offset = page_id * self.page_size
            page = self._pack_page(
                full_host_indices[token_offset].item(),
                swa_host_indices[token_offset].item(),
            )
            for chunk_id in range(self.page_chunk_count):
                chunk_offset = chunk_id * self.eic_chunk_bytes
                chunk_end = min(chunk_offset + self.eic_chunk_bytes, self.page_bytes)
                dst = dst_tensors[page_id * self.page_chunk_count + chunk_id].view(
                    torch.uint8
                )
                dst.zero_()
                if chunk_offset < chunk_end:
                    dst[: chunk_end - chunk_offset].copy_(
                        page[chunk_offset:chunk_end], non_blocking=False
                    )

    def device_writeback(self, device_indices, src_tensor, src_indices):
        chunk_count = len(src_indices)
        page_count = chunk_count // self.page_chunk_count
        if page_count == 0:
            return
        device_indices = device_indices[: page_count * self.page_size]
        full_host_indices, swa_host_indices, transfers = self._build_pool_transfers(
            device_indices
        )

        for page_id in range(page_count):
            pieces = []
            for chunk_id in range(self.page_chunk_count):
                src_index = src_indices[page_id * self.page_chunk_count + chunk_id]
                chunk_offset = chunk_id * self.eic_chunk_bytes
                chunk_end = min(chunk_offset + self.eic_chunk_bytes, self.page_bytes)
                pieces.append(
                    src_tensor[src_index]
                    .view(torch.uint8)[: chunk_end - chunk_offset]
                    .contiguous()
                )
            page_data = torch.cat(pieces)
            token_offset = page_id * self.page_size
            self._unpack_page(
                full_host_indices[token_offset].item(),
                swa_host_indices[token_offset].item(),
                page_data,
            )
        torch.cuda.current_stream().synchronize()

        # load_to_device_per_layer takes indices as given (backup normalizes,
        # load does not), so move host indices to the device for the kernel
        # backend. The new API is per-pool: the group-level entry point is gone.
        if self.io_backend == "kernel":
            for transfer in transfers:
                transfer.host_indices = transfer.host_indices.to(
                    transfer.device_indices.device
                )

        for layer_id in range(self.transfer_layer_num):
            for transfer in transfers:
                entry = self.host_pool_group.get_entry(transfer.name)
                local_layer_id = (
                    entry.layer_mapper(layer_id)
                    if entry.layer_mapper is not None
                    else layer_id
                )
                if local_layer_id is None:
                    continue
                entry.host_pool.load_to_device_per_layer(
                    entry.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    local_layer_id,
                    self.io_backend,
                )
        torch.cuda.current_stream().synchronize()

    def assign_page_data(self, content_hashes, flat_data, device_indices=None):
        logger.debug(f"assign_deepseek_v4_page_data hashes {content_hashes}")
        if len(content_hashes) == 0:
            return True
        if device_indices is None:
            logger.error("DeepSeek V4 EIC assign_page_data requires device_indices")
            return False
        page_batch_size = max(1, G_GDRBounceTensorCount // self.page_chunk_count)
        for i in range(0, len(content_hashes), page_batch_size):
            page_hashes = content_hashes[i : i + page_batch_size]
            key = self._encode_page_chunk_keys(page_hashes)
            indices = device_indices[
                i * self.page_size : (i + len(page_hashes)) * self.page_size
            ]
            if G_EnableAsyncKVSet:
                ret = self.eic_client.async_batch_set(
                    key, None, indices, copy_func=self.device_backup
                )
            else:
                ret = self.eic_client.set(
                    key, None, indices, copy_func=self.device_backup
                )
            if not ret:
                logger.error(
                    f"assign_deepseek_v4_page_data keys {key} failed, eic_client return none"
                )
                return False
        return True

    def get_page_data(self, content_hashs, device_indices=None):
        logger.debug(f"get_deepseek_v4_page_data content_hashs {content_hashs}")
        if device_indices is None:
            logger.error("DeepSeek V4 EIC get_page_data requires device_indices")
            return [False] * len(content_hashs)
        page_batch_size = max(1, G_GDRBounceTensorCount // self.page_chunk_count)
        success_mask = []
        for i in range(0, len(content_hashs), page_batch_size):
            page_hashes = content_hashs[i : i + page_batch_size]
            key = self._encode_page_chunk_keys(page_hashes)
            indices = device_indices[
                i * self.page_size : (i + len(page_hashes)) * self.page_size
            ]
            _, chunk_mask = self.eic_client.batch_get(
                key, indices, copy_func=self.device_writeback
            )
            page_mask = []
            for page_id in range(len(page_hashes)):
                start = page_id * self.page_chunk_count
                end = start + self.page_chunk_count
                chunk = chunk_mask[start:end]
                # all([]) is True: a truncated reply would read as a full hit.
                page_mask.append(len(chunk) == self.page_chunk_count and all(chunk))
            success_mask.extend(page_mask)
            if not all(page_mask):
                break
        return success_mask


if __name__ == "__main__":
    # Example usage
    kv_cache_shape = (128, 2048)  # Example shape
    numel = 1
    for i in kv_cache_shape:
        numel *= i
    kv_cache_dtype = torch.float32  # Example dtype

    import sys

    if len(sys.argv) > 1:
        data_device, mem_pool_device = sys.argv[1].split(",")
    else:
        data_device, mem_pool_device = "cpu", "cpu"

    eic_client = EICKVClient(kv_cache_dtype, kv_cache_shape, mem_pool_device)
    print(
        f"EIC Client initialized with data_device: {data_device}, mem_pool_device: {mem_pool_device}, kv_cache_shape: {kv_cache_shape}, kv_cache_dtype: {kv_cache_dtype}"
    )

    test_keys = 1024
    batch_size = 256
    keys = ["test_key_" + str(i) for i in range(test_keys)]
    values = [
        torch.randn(kv_cache_shape, dtype=kv_cache_dtype, device=data_device)
        for i in range(test_keys)
    ]

    last_time = time.perf_counter()
    transmit_bytes = 0
    write_size = test_keys * numel * kv_cache_dtype.itemsize

    total_round = 100
    for round_i in range(total_round):
        for i in range(0, test_keys, batch_size):
            eic_client._sync_set_impl(
                keys[i : i + batch_size], values[i : i + batch_size]
            )
        transmit_bytes += write_size
        if transmit_bytes >= 2 * 1024 * 1024 * 1024:
            cost_time = time.perf_counter() - last_time
            print(
                f"data_device: {data_device}, mem_pool_device: {mem_pool_device}, write {transmit_bytes} bytes, cost time {cost_time}, bps {transmit_bytes // cost_time // 1024 // 1024} MB/s"
            )
            last_time = time.perf_counter()
            transmit_bytes = 0

    get_stream = torch.cuda.current_stream()
    for round_i in range(total_round):
        for i in range(0, test_keys, batch_size):
            recv_objs, success = eic_client.batch_get(keys[i : i + batch_size])
            with torch.cuda.stream(get_stream):
                for j, obj in enumerate(recv_objs):
                    obj.copy_(values[i + j], non_blocking=True)
            get_stream.synchronize()

        transmit_bytes += write_size
        if transmit_bytes >= 2 * 1024 * 1024 * 1024:
            cost_time = time.perf_counter() - last_time
            print(
                f"data_device: {data_device}, mem_pool_device: {mem_pool_device}, read {transmit_bytes} bytes, cost time {cost_time}, bps {transmit_bytes // cost_time // 1024 // 1024} MB/s"
            )
            last_time = time.perf_counter()
            transmit_bytes = 0

    del eic_client  # Clean up the client to release resources
    print("EIC Client cleaned up.")

import os
import time
import torch
from torch import Tensor
from typing import Dict, Optional, Set, Tuple, Union
import threading

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    BaseTokenToKVPool,
    MHATokenToKVPool
)

from hip.models.hip_attention.gen3.uvm_gpu_cache import (
    UVMCache,
    GPUCache,
    HiPOffloadCache,
    format_size_bytes,
)
import logging
from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig

logger = logging.getLogger(__name__)

class MHATokenToHiPOffloadKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        max_token_size: int,
        max_mask_cache_token_size: int,
        max_sa_cache_token_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device,
        hip_config: HiPAttentionConfig,
    ):
        assert isinstance(device, torch.device)
        assert device.index is not None
        super().__init__(max_token_size, dtype, device)
        
        #TODO: derive token sizes from size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.max_mask_cache_token_size = max_mask_cache_token_size * head_num
        self.max_sa_cache_token_size = max_sa_cache_token_size * head_num
        
        self.online_update_cache = os.getenv('DEBUG_ONLINE', '0') == '1'
        self.layer_buffer = []
        for layer_id in range(layer_num):
            self.layer_buffer.append(
                HiPOffloadCache(
                    layer_id=layer_id,
                    max_token_size=max_token_size + 1,
                    max_mask_cache_token_size=min(max_token_size * head_num, self.max_mask_cache_token_size),
                    max_sa_cache_token_size=min(max_token_size * head_num, self.max_sa_cache_token_size),
                    head_num=head_num,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    online_cache_update=self.online_update_cache,
                )
                if layer_id not in hip_config.dense_layers else
                HiPOffloadCache(
                    layer_id=layer_id,
                    max_token_size=max_token_size + 1,
                    max_mask_cache_token_size=min(max_token_size * head_num, self.max_mask_cache_token_size * 2),
                    max_sa_cache_token_size=min(max_token_size * head_num, self.max_sa_cache_token_size * 2),
                    head_num=head_num,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    online_cache_update=self.online_update_cache,
                )
            )
            
            uvm_allocated_bytes, gpu_allocated_bytes = self.calc_allocated_bytes()
            logger.info(
                f'[{layer_id + 1}/{layer_num}] '
                f'Allocated total CPU (UVM) bytes: {format_size_bytes(uvm_allocated_bytes)}, '
                f'Allocated total GPU bytes: {format_size_bytes(gpu_allocated_bytes)}, '
                f'{self.dtype} on {self.device}'
            )
        
        # (layer_id, batch_id) -> (K, V, seq_len)
        self.prefetch_threads: Dict[Tuple[int, int], threading.Thread] = {}
        self.prefetched_kv: Dict[Tuple[int, int], Tuple[Tensor, Tensor, int]] = {}
        
        self.async_set_threads: Set[threading.Thread] = set()
        
        self.enable_async = os.getenv('HIP_DISABLE_AYSNC', '0') == '0'
        
        # uvm_allocated_bytes, gpu_allocated_bytes = self.calc_allocated_bytes()
        # logger.info(
        #     f'Allocated total CPU (UVM) bytes: {format_size_bytes(uvm_allocated_bytes)}, '
        #     f'Allocated total GPU bytes: {format_size_bytes(gpu_allocated_bytes)}, '
        #     f'{self.dtype} on {self.device}'
        # )

        self.require_validation = os.getenv('HIP_OFFLOAD_CACHE_VALIDATION', '0') == '1'
        if self.require_validation:
            self.validation_cache = MHATokenToKVPool(
                max_token_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=layer_num,
                device=self.device,
            )
        else:
            self.validation_cache = None
    
    def calc_allocated_bytes(self):
        uvm_allocated_bytes = 0
        gpu_allocated_bytes = 0
        for cache in self.layer_buffer:
            uvm_allocated_bytes += cache.k_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.k_uvm.allocated_gpu_bytes
            uvm_allocated_bytes += cache.v_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.v_uvm.allocated_gpu_bytes
            gpu_allocated_bytes += cache.mask_k_cache.allocated_gpu_bytes
            gpu_allocated_bytes += cache.sa_kv_cache.allocated_gpu_bytes
        return uvm_allocated_bytes, gpu_allocated_bytes

    def get_key_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> HiPOffloadCache:
        # Use this function for decode, pass this to `k`
        if self.require_validation:
            return self.layer_buffer[layer_id], *self.validation_cache.get_kv_buffer(layer_id)
        return self.layer_buffer[layer_id]
    
    def prefetch_prefix_kv_buffer(
        self, 
        layer_id: int, 
        batch_id: int, 
        table: Tensor, 
        prefix_seq_len: int
    ) -> threading.Thread:
        # you must call before get fetched prefix
        assert table.ndim == 1
        
        if self.require_validation:
            hip_offload_cache, _, _ = self.get_kv_buffer(layer_id)
        else:
            hip_offload_cache = self.get_kv_buffer(layer_id)
        
        handle_id = (layer_id, batch_id)
        assert handle_id not in self.prefetch_threads, handle_id
        assert handle_id not in self.prefetched_kv, handle_id


        if self.enable_async:
            start_event = torch.cuda.Event()
            table = table.to(torch.int64).to('cpu')
            start_event.record()
            # torch.cuda.synchronize()
            def thread_main():
                try:
                    # BUG(heejun): i think this line is quite suspicious hmm
                    start_event.synchronize()
                    stream = torch.cuda.Stream(device=self.device, priority=0)

                    with torch.cuda.stream(stream):
                        k, v = hip_offload_cache.prefetch_prefix_kv_buffer(
                            table=table,
                            device=self.device,
                        )
                        assert k.device == self.device
                        assert v.device == self.device
                    
                    stream.synchronize()
                    self.prefetched_kv[handle_id] = (k, v, prefix_seq_len, table)
                except Exception as ex:
                    print(f'{handle_id} thread dead')
                    raise Exception('thread dead') from ex
                finally:
                    self.prefetch_threads.pop(handle_id)
            
            t = threading.Thread(target=thread_main, daemon=True)
            self.prefetch_threads[handle_id] = t
            t.start()
        else:
            k, v = hip_offload_cache.prefetch_prefix_kv_buffer(
                table=table.to(torch.int64),
                device=self.device,
            )
            assert k.device == self.device
            assert v.device == self.device
            torch.cuda.synchronize()
            self.prefetched_kv[handle_id] = (k, v, prefix_seq_len, table)
        return
    
    def get_fetched_prefix_kv_buffer(
        self,
        layer_id: int,
        batch_id: int,
        # you need to pass KV for extend
        cache_k: Tensor,
        cache_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # return cache_k, cache_v
    
        # Use this function for prefill
        handle_id = (layer_id, batch_id)
        prefetch_thread = self.prefetch_threads.get(handle_id, None)
        if prefetch_thread is not None:
            while handle_id not in self.prefetched_kv:
                time.sleep(0.0001)
            # print('start join', flush=True)
            # while True:
            #     try:
            #         prefetch_thread.join(timeout=1.0)
            #         print('joined')
            #         break
            #     except TimeoutError:
            #         print('timeout', layer_id, batch_id)
            #     except RuntimeError:
            #         print('runtime error wtf')
            #         raise RuntimeError('deadlock')
        
        assert handle_id in self.prefetched_kv, "did prefetch successed?"
        k, v, prefix_seq_len, table = self.prefetched_kv.pop(handle_id)
        
        assert isinstance(k, Tensor)
        assert isinstance(v, Tensor)
        assert isinstance(prefix_seq_len, int)
        assert k.shape == v.shape
        assert k.ndim == 4, f'{k.shape}'
        assert k.shape[0] == 1
        assert k.shape[1] >= prefix_seq_len
        assert k.shape[2] == self.head_num
        assert k.shape[3] == self.head_dim
        assert k.dtype == v.dtype
        assert k.dtype == self.dtype
        assert cache_k.ndim == 4
        assert cache_k.shape[0] == 1
        assert cache_k.shape[2] == self.head_num
        assert cache_k.shape[3] == self.head_dim
        assert k.shape[1] == prefix_seq_len + cache_k.shape[1]
        assert k.dtype in [torch.float8_e5m2, torch.float16, torch.bfloat16, torch.float32]

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        # if self.dtype not in [torch.float8_e5m2]:
        #     assert cache_k.dtype == self.dtype
        # else:
        #     if cache_k.dtype != self.dtype:
        #         cache_k = cache_k.to(self.dtype)
        #         cache_v = cache_v.to(self.dtype)
        
        k[:, prefix_seq_len:, :, :] = cache_k
        v[:, prefix_seq_len:, :, :] = cache_v

        if self.require_validation:
            k_valid, v_valid = self.validation_cache.get_kv_buffer(layer_id)

            assert k.dtype == k_valid.dtype

            k_valid_packed = k_valid[table].unsqueeze(0)
            v_valid_packed = v_valid[table].unsqueeze(0)

            k_err = ((k_valid_packed - k) ** 2).sum()
            v_err = ((v_valid_packed - v) ** 2).sum()

            assert k_err < 1e-5, k_err
            assert v_err < 1e-5, v_err

            return k, v, k_valid, v_valid
        else:
            return k, v

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        async_copy: bool = False,
        push_to_gpu_cache: bool = False,
    ):
        if self.require_validation:
            self.validation_cache.set_kv_buffer(
                layer, table, cache_k, cache_v,
            )

        if not self.enable_async:
            async_copy = False
        
        layer_id = layer.layer_id
        # pass async_copy=True when only prefill (eager mode)
        assert (not async_copy) or (async_copy and (not torch.cuda.is_current_stream_capturing()))
        
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        
        if async_copy:
            start_event = torch.cuda.Event()
            start_event.record()

            def thread_main():
                try:
                    start_event.synchronize()
                    stream = torch.cuda.Stream(device=self.device)

                    with torch.cuda.stream(stream):
                        table_gpu = table.to(torch.int64)
                        table_cpu = table.to('cpu', non_blocking=False)
                        cache_k_cpu = cache_k.to('cpu', non_blocking=False)
                        cache_v_cpu = cache_v.to('cpu', non_blocking=False)
                        self.layer_buffer[layer_id].set_kv_buffer(
                            table=table_cpu,
                            table_gpu=table_gpu,
                            cache_k=cache_k_cpu,
                            cache_v=cache_v_cpu,
                        )
                    stream.synchronize()
                finally:
                    self.async_set_threads.remove(t)
            
            t = threading.Thread(target=thread_main, daemon=True)
            self.async_set_threads.add(t)
            t.start()
        else:
            self.layer_buffer[layer_id].set_kv_buffer(
                table=table,
                table_gpu=table,
                cache_k=cache_k,
                cache_v=cache_v,
            )
    
    def synchronize(self):
        torch.cuda.synchronize(device=self.device)
        t = time.time()
        # you must call this function when finish prefill, before decode
        while (len(self.prefetch_threads) > 0) or (len(self.async_set_threads) > 0):
            time.sleep(0.001)
        assert len(self.prefetch_threads) == 0
        assert len(self.async_set_threads) == 0
        assert len(self.prefetched_kv) == 0
        elapsed = time.time() - t
        logger.debug(f'Final layer sync took {elapsed * 1024:.4f} ms')
    
    def prefetch_layer(self, forward_batch: ForwardBatch, layer_id: int):
        assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
        assert forward_batch.token_to_kv_pool == self

        for ibatch in range(forward_batch.batch_size):
            req_to_tokens = forward_batch.req_to_token_pool.req_to_token
            req_pool_indices = forward_batch.req_pool_indices[ibatch:ibatch+1]
            block_table = req_to_tokens.index_select(
                dim=0, index=req_pool_indices
            )[0, :forward_batch.extend_prefix_lens_cpu[ibatch] + forward_batch.extend_seq_lens_cpu[ibatch]]
            # print(block_table, block_table.shape)
            self.prefetch_prefix_kv_buffer(
                layer_id=layer_id, 
                batch_id=ibatch,
                table=block_table,
                prefix_seq_len=forward_batch.extend_prefix_lens_cpu[ibatch]
            )
    
    def wait_prefetch_layer(self, forward_batch: ForwardBatch, layer_id: int):
        assert isinstance(forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool)
        assert forward_batch.token_to_kv_pool == self

        for ibatch in range(forward_batch.batch_size):
            while (layer_id, ibatch) not in self.prefetched_kv:
                time.sleep(0.0001)
    
    def on_model_start(self, forward_batch: ForwardBatch):
        require_prefetch = forward_batch.forward_mode.is_extend()
        assert forward_batch.token_to_kv_pool == self

        if require_prefetch:
            # FIXME: find better way to detect this.
            is_first_chunk = forward_batch.extend_prefix_lens_cpu[0] == 0
            # FIXME: find better way to detect this.
            is_inter_chunk = forward_batch.extend_seq_lens_cpu[0] in map(lambda x: 2**x, range(0, 20))
            # BUG(heejun): at the last chunk of prefill, prefetch layer sometimes failes... so disable async
            if not (forward_batch.batch_size == 1 and (is_first_chunk or is_inter_chunk)):
                self.onetime_disable = self.enable_async
                self.enable_async = False
            self.prefetch_layer(forward_batch, 0)
            # self.wait_prefetch_layer(forward_batch, 0)

    def on_model_end(self, forward_batch: ForwardBatch):
        require_prefetch = forward_batch.forward_mode.is_extend()
        assert forward_batch.token_to_kv_pool == self
        
        if require_prefetch:
            self.synchronize()
            self.enable_async = self.enable_async or self.onetime_disable
            self.onetime_disable = False

    def on_layer_start(self, forward_batch: ForwardBatch, layer_id: int):
        require_prefetch = forward_batch.forward_mode.is_extend()
        assert forward_batch.token_to_kv_pool == self

        if require_prefetch and (layer_id < (self.layer_num - 1)):
            self.prefetch_layer(forward_batch, layer_id + 1)

    def on_layer_end(self, forward_batch: ForwardBatch, layer_id: int):
        require_prefetch = forward_batch.forward_mode.is_extend()
        assert forward_batch.token_to_kv_pool == self

        if require_prefetch and (layer_id < (self.layer_num - 1)):
            torch.cuda.current_stream(self.device).synchronize()
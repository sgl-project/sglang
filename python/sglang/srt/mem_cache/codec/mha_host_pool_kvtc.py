from __future__ import annotations

import logging

import torch

from sglang.srt.mem_cache.codec.kvtc_page_codec import KvtcPageCodec
from sglang.srt.mem_cache.codec.kv_page_meta import KVPageMeta
from sglang.srt.mem_cache.codec.l2_codec_config import HiCacheL2CodecConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import HostKVCache, synchronized
from sglang.srt.utils.common import get_bool_env_var, get_int_env_var

logger = logging.getLogger(__name__)


class MHATokenToKVPoolHostKVTC(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        *,
        l2_codec: HiCacheL2CodecConfig,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        if not l2_codec.params_path:
            raise ValueError("kvtc params_path is required for L2 codec")
        self.codec_name = "kvtc"
        self.kv_order = "k_all_v_all"
        self._l2_codec_cfg = l2_codec
        self._codec = KvtcPageCodec(l2_codec.params_path)
        self._compressed_pages: dict[int, bytes] = {}
        self._page_spans: dict[int, tuple[int, int, int]] = {}
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )

        if self.layout != "layer_first":
            raise ValueError(
                f"MHATokenToKVPoolHostKVTC only supports layout='layer_first', got {self.layout}"
            )

        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.v_head_dim = self.device_pool.v_head_dim
        self.layer_num = self.device_pool.layer_num
        self._bytes_per_k_token = (
            self.head_num * self.head_dim * self.dtype.itemsize
        )
        self._bytes_per_v_token = (
            self.head_num * self.v_head_dim * self.dtype.itemsize
        )
        self._page_numel = (
            self.layer_num
            * self.page_size
            * self.head_num
            * (self.head_dim + self.v_head_dim)
        )
        self._per_layer_k = self.page_size * self.head_num * self.head_dim
        self._per_layer_v = self.page_size * self.head_num * self.v_head_dim
        self._encode_page_buffer = torch.empty(
            (self._page_numel,), dtype=torch.float32, device="cpu"
        )
        self._encode_page_buffer_gpu = None
        self._encode_page_buffer_gpu_device = None
        self._k_offsets = [
            layer_id * self._per_layer_k for layer_id in range(self.layer_num)
        ]
        self._v_offsets = [
            self.layer_num * self._per_layer_k + layer_id * self._per_layer_v
            for layer_id in range(self.layer_num)
        ]
        self._kv_page_meta = KVPageMeta(
            page_size=self.page_size,
            layout=self.layout,
            is_mla_model=False,
            tp_rank=0,
            tp_size=1,
            model_name=None,
        ).to_dict()
        self._sink_tokens = get_int_env_var("SGLANG_KVTC_SINK_TOKENS", 4)
        self._recent_tokens = get_int_env_var("SGLANG_KVTC_RECENT_TOKENS", 128)
        self._bypass_log = get_bool_env_var("SGLANG_KVTC_BYPASS_LOG", "false")
        self._bypass_log_max_pages = get_int_env_var("SGLANG_KVTC_BYPASS_LOG_MAX_PAGES", 8)

        logger.info(
            "HiCache L2 codec enabled: kvtc(params=%s), page_size=%d",
            l2_codec.params_path,
            self.page_size,
        )

    def get_size_per_token(self):
        head_num = self.device_pool.head_num
        head_dim = self.device_pool.head_dim
        v_head_dim = self.device_pool.v_head_dim
        layer_num = self.device_pool.layer_num
        raw = (
            head_num * (head_dim + v_head_dim) * layer_num * self.dtype.itemsize
        )
        ratio = max(float(self._l2_codec_cfg.ratio), 1.0)
        return max(int(raw / ratio), 1)

    def init_kv_buffer(self):
        return torch.empty((0,), dtype=torch.uint8, device=self.device)

    @synchronized
    def clear(self):
        super().clear()
        self._compressed_pages.clear()
        self._page_spans.clear()

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.detach().cpu().tolist()
        for i in range(0, len(indices_cpu), self.page_size):
            base = int(indices_cpu[i])
            self._compressed_pages.pop(base, None)
            self._page_spans.pop(base, None)
        return super().free(indices)

    @synchronized
    def set_page_span(self, host_base: int, abs_start: int, abs_end: int, node_abs_end: int) -> None:
        self._page_spans[int(host_base)] = (int(abs_start), int(abs_end), int(node_abs_end))

    def _unpack_page_vector(self, vec: torch.Tensor, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        k_offset = self._k_offsets[layer_id]
        v_offset = self._v_offsets[layer_id]
        k = vec[k_offset : k_offset + self._per_layer_k]
        v = vec[v_offset : v_offset + self._per_layer_v]
        k = k.reshape(self.page_size, self.head_num, self.head_dim)
        v = v.reshape(self.page_size, self.head_num, self.v_head_dim)
        return k, v

    def _get_encode_page_buffer(self, device: torch.device) -> torch.Tensor:
        if device.type == "cuda" and getattr(self._codec, "_gpu_project_enabled", False):
            if (
                self._encode_page_buffer_gpu is None
                or self._encode_page_buffer_gpu_device != device
            ):
                self._encode_page_buffer_gpu = torch.empty(
                    (self._page_numel,), dtype=torch.float32, device=device
                )
                self._encode_page_buffer_gpu_device = device
            return self._encode_page_buffer_gpu
        return self._encode_page_buffer

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        blob = self._compressed_pages.get(int(index))
        if blob is None:
            data = torch.zeros((self._page_numel,), dtype=self.dtype, device="cpu")
            return data if flat else data
        vec = self._codec.decode_vector(blob, self._kv_page_meta).to(self.dtype)
        return vec.flatten() if flat else vec

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros((self._page_numel,), dtype=self.dtype, device="cpu")

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        vec = data_page.view(-1).to(torch.float32).cpu()
        blob = self._codec.encode_vector(vec, self._kv_page_meta)
        self._compressed_pages[int(index)] = blob

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_list = host_indices.detach().cpu().tolist()
        dev_list = device_indices.detach().cpu().tolist()
        if len(host_list) != len(dev_list):
            raise ValueError("host_indices and device_indices must have same length")
        if len(host_list) % self.page_size != 0:
            raise ValueError("indices length must be multiple of page_size")

        for i in range(0, len(host_list), self.page_size):
            host_base = int(host_list[i])
            locs = dev_list[i : i + self.page_size]
            device = device_pool.k_buffer[0].device
            vec = self._get_encode_page_buffer(device)
            for layer_id in range(self.layer_num):
                k = device_pool.k_buffer[layer_id][locs].contiguous().view(-1)
                v = device_pool.v_buffer[layer_id][locs].contiguous().view(-1)
                vec[
                    self._k_offsets[layer_id] : self._k_offsets[layer_id]
                    + self._per_layer_k
                ].copy_(k.to(device=vec.device, dtype=torch.float32))
                vec[
                    self._v_offsets[layer_id] : self._v_offsets[layer_id]
                    + self._per_layer_v
                ].copy_(v.to(device=vec.device, dtype=torch.float32))
            kv_page_meta = dict(self._kv_page_meta)
            span = self._page_spans.get(host_base)
            if span is not None:
                abs_start, abs_end, node_abs_end = span
                kv_page_meta["abs_start"] = abs_start
                kv_page_meta["abs_end"] = abs_end
                kv_page_meta["node_abs_end"] = node_abs_end
                bypass = False
                bypass_reason = None
                if self._sink_tokens > 0 and abs_start < self._sink_tokens:
                    bypass = True
                    bypass_reason = "sink"
                if self._recent_tokens > 0 and abs_end > max(node_abs_end - self._recent_tokens, 0):
                    bypass = True
                    bypass_reason = "recent"
                if bypass:
                    kv_page_meta["bypass"] = True
                if self._bypass_log:
                    page_idx = i // self.page_size
                    if page_idx < self._bypass_log_max_pages:
                        logger.info(
                            "KVTC bypass: host_base=%s abs=[%d,%d) node_abs_end=%d bypass=%s reason=%s",
                            host_base,
                            abs_start,
                            abs_end,
                            node_abs_end,
                            bypass,
                            bypass_reason,
                        )

            blob = self._codec.encode_vector(vec, kv_page_meta)
            self._compressed_pages[host_base] = blob

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        host_list = host_indices.detach().cpu().tolist()
        dev_list = device_indices.detach().cpu().tolist()
        if len(host_list) != len(dev_list):
            raise ValueError("host_indices and device_indices must have same length")
        if len(host_list) % self.page_size != 0:
            raise ValueError("indices length must be multiple of page_size")

        for i in range(0, len(host_list), self.page_size):
            host_base = int(host_list[i])
            locs = dev_list[i : i + self.page_size]
            blob = self._compressed_pages.get(host_base)
            if blob is None:
                continue
            vec = self._codec.decode_vector(blob, self._kv_page_meta)
            k, v = self._unpack_page_vector(vec, layer_id)
            device_pool.k_buffer[layer_id][locs] = k.to(
                device_pool.k_buffer[layer_id].device
            )
            device_pool.v_buffer[layer_id][locs] = v.to(
                device_pool.v_buffer[layer_id].device
            )

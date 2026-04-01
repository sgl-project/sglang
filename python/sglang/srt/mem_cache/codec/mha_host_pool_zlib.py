from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.codec.l2_codec_config import HiCacheL2CodecConfig
from sglang.srt.mem_cache.codec.zlib_page_codec import ZlibCodecConfig, ZlibPageCodec
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import HostKVCache, synchronized

logger = logging.getLogger(__name__)


def _uint8_tensor_to_bytes(t: torch.Tensor) -> bytes:
    t = t.contiguous().reshape(-1).cpu()
    try:
        import numpy as np  # type: ignore

        return np.asarray(t).tobytes()
    except Exception:
        return bytes(t.tolist())


def _bytes_to_uint8_tensor(buf: bytes) -> torch.Tensor:
    if hasattr(torch, "frombuffer"):
        return torch.frombuffer(memoryview(buf), dtype=torch.uint8)
    try:
        import numpy as np  # type: ignore

        return torch.as_tensor(np.frombuffer(buf, dtype=np.uint8))
    except Exception:
        return torch.tensor(list(buf), dtype=torch.uint8)


class MHATokenToKVPoolHostZlib(HostKVCache):
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
        self.codec_name = "zlib"
        self.kv_order = "interleaved_per_layer"
        self._l2_codec_cfg = l2_codec
        self._codec = ZlibPageCodec(config=ZlibCodecConfig(level=l2_codec.level))
        self._compressed_pages: dict[int, bytes] = {}
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
                f"MHATokenToKVPoolHostZlib only supports layout='layer_first', got {self.layout}"
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
        self._bytes_per_k_page = self._bytes_per_k_token * self.page_size
        self._bytes_per_v_page = self._bytes_per_v_token * self.page_size

        logger.info(
            "HiCache L2 codec enabled: zlib(level=%s, ratio=%.3f), page_size=%d",
            l2_codec.level,
            l2_codec.ratio,
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

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.detach().cpu().tolist()
        for i in range(0, len(indices_cpu), self.page_size):
            base = int(indices_cpu[i])
            self._compressed_pages.pop(base, None)
        return super().free(indices)

    def _get_page_bytes(self, index: int) -> Optional[bytes]:
        blob = self._compressed_pages.get(int(index))
        if blob is None:
            return None
        return self._codec.decode_bytes(blob)

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raw = self._get_page_bytes(int(index))
        if raw is None:
            data = torch.zeros(
                (
                    self.layer_num,
                    self.page_size,
                    self._bytes_per_k_token + self._bytes_per_v_token,
                ),
                dtype=torch.uint8,
                device="cpu",
            ).flatten()
            return data if flat else data

        u8 = _bytes_to_uint8_tensor(raw)
        return u8.flatten() if flat else u8

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                self._bytes_per_k_token + self._bytes_per_v_token,
            ),
            dtype=torch.uint8,
            device="cpu",
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        blob = self._codec.encode_bytes(_uint8_tensor_to_bytes(data_page.view(torch.uint8)))
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
            parts = []
            for layer_id in range(self.layer_num):
                k_u8 = (
                    device_pool.k_buffer[layer_id][locs]
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                )
                v_u8 = (
                    device_pool.v_buffer[layer_id][locs]
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                )
                parts.append(_uint8_tensor_to_bytes(k_u8))
                parts.append(_uint8_tensor_to_bytes(v_u8))
            self._compressed_pages[host_base] = self._codec.encode_bytes(b"".join(parts))

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
            raw = self._get_page_bytes(host_base)
            if raw is None:
                continue
            u8 = _bytes_to_uint8_tensor(raw)
            offset = 0
            for l in range(layer_id):
                offset += self._bytes_per_k_page + self._bytes_per_v_page
            k_bytes = u8[offset : offset + self._bytes_per_k_page]
            v_bytes = u8[
                offset + self._bytes_per_k_page : offset + self._bytes_per_k_page + self._bytes_per_v_page
            ]

            k = (
                k_bytes.to(device_pool.k_buffer[layer_id].device)
                .view(self.dtype)
                .reshape(self.page_size, self.head_num, self.head_dim)
            )
            v = (
                v_bytes.to(device_pool.v_buffer[layer_id].device)
                .view(self.dtype)
                .reshape(self.page_size, self.head_num, self.v_head_dim)
            )
            device_pool.k_buffer[layer_id][locs] = k
            device_pool.v_buffer[layer_id][locs] = v

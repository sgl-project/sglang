
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseekv4_memory_pool import get_compress_state_ring_size

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class DSv4PoolSizes:

    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSv4MemoryCalculator:

    def __init__(
        self,
        model_config: ModelConfig,
        page_size: int,
        swa_ratio: float,
        is_speculative: bool = False,
        c4_shrink_factor: int = 1,
    ):
        self.qk_nope_head_dim = model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_config.qk_rope_head_dim
        self.indexer_head_dim = model_config.index_head_dim
        self.compression_ratios = model_config.compress_ratios
        self.swa_page_size = model_config.window_size
        self.page_size = page_size
        self.swa_ratio = swa_ratio
        self.is_speculative = is_speculative
        assert c4_shrink_factor >= 1
        self.c4_shrink_factor = c4_shrink_factor

        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        self.bytes_per_full_token = self.get_bytes_per_full_token()

    def get_bytes_per_full_token(self) -> float:
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        state_dtype_size = 4
        c4_state_bytes = 2 * 2 * attn_head_dim * state_dtype_size
        # Online c128 stores (max, sum, kv) per slot (3*head_dim) instead of
        # raw (kv, score) (2*head_dim). Combined with ring_size=1 this still
        # nets a large reduction (~3/256x) but the per-slot bytes go up.
        c128_online = envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        c128_state_bytes = (3 if c128_online else 2 * 1) * attn_head_dim * state_dtype_size
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * state_dtype_size

        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        c128_state_ratio = self.c128_ring_size / self.swa_page_size

        c4_frac = 1 / (4 * self.c4_shrink_factor)
        bytes_per_full_token = (
            self.swa_ratio * kv_bytes * self.num_layers_total
            + c4_frac * kv_bytes * self.num_layers_ca4
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            + self.swa_ratio
            * c128_state_ratio
            * c128_state_bytes
            * self.num_layers_ca128
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

        return bytes_per_full_token

    def calculate_pool_sizes(self, available_bytes: int) -> DSv4PoolSizes:
        full_token = int(available_bytes / self.bytes_per_full_token)

        full_token = full_token // self.page_size * self.page_size

        swa_tokens = int(full_token * self.swa_ratio) // self.page_size * self.page_size

        pool_sizes = DSv4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // (4 * self.c4_shrink_factor),
            c128_max_total_num_tokens=full_token // 128,
            c4_state_pool_size=swa_tokens // self.swa_page_size * self.c4_ring_size,
            c128_state_pool_size=swa_tokens // self.swa_page_size * self.c128_ring_size,
        )

        logger.info(
            f"DSv4 memory calculation: "
            f"bytes_per_full_token={self.bytes_per_full_token:.2f}, "
            f"available_bytes={available_bytes / (1 << 30):.2f} GB, "
            f"full_token={full_token}"
        )

        return pool_sizes

    def get_pool_sizes_by_profiling(self, mr: ModelRunner) -> DSv4PoolSizes:
        available_bytes = profile_available_bytes(
            device=mr.device,
            gpu_id=mr.gpu_id,
            total_gpu_memory=mr.total_gpu_memory,
            mem_fraction_static=mr.mem_fraction_static,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )

        if self.is_speculative:
            draft_layers = 1
            target_layers = self.num_layers_total
            target_ratio = target_layers / (target_layers + draft_layers)
            available_bytes = int(available_bytes * target_ratio)

        return self.calculate_pool_sizes(available_bytes)

    def get_pool_sizes_by_configuration(self, max_total_tokens: int) -> DSv4PoolSizes:
        available_bytes = max_total_tokens * self.bytes_per_full_token
        return self.calculate_pool_sizes(available_bytes)


def profile_available_bytes(
    device: str,
    gpu_id: int,
    total_gpu_memory: float,
    mem_fraction_static: float,
    distributed: bool = False,
    cpu_group=None,
) -> int:
    from sglang.srt.utils.common import get_available_gpu_memory

    available_gpu_memory = get_available_gpu_memory(
        device, gpu_id, distributed=distributed, cpu_group=cpu_group
    )
    rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)

    available_bytes = int(rest_memory * (1 << 30))

    logger.info(
        f"Memory profiling: available_gpu_memory={available_gpu_memory:.2f} GB, "
        f"total_gpu_memory={total_gpu_memory:.2f} GB, "
        f"mem_fraction_static={mem_fraction_static:.2f}, "
        f"rest_memory={rest_memory:.2f} GB"
    )

    return available_bytes

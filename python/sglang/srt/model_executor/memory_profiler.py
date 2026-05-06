"""Memory profiler for DeepSeekV4 and other models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.mem_cache.deepseekv4_memory_pool import get_compress_state_ring_size

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class DSv4PoolSizes:
    """Pool sizes for DeepSeekV4 memory allocation."""

    full_max_total_num_tokens: int
    swa_max_total_num_tokens: int
    c4_max_total_num_tokens: int
    c128_max_total_num_tokens: int
    c4_state_pool_size: int
    c128_state_pool_size: int


class DSv4MemoryCalculator:
    """Calculate pool sizes for DeepSeekV4 memory allocation.

    Memory pools for DSv4:
    - SWA KV pool: size=F*R, layers=num_layers_total
    - C4 KV pool: size=F/4, layers=num_layers_ca4
    - C128 KV pool: size=F/128, layers=num_layers_ca128
    - C4 Indexer pool: size=F/4, layers=num_layers_ca4
    - C4 State pool (paged): size=F*R/swa_page_size*c4_ring_size, layers=num_layers_ca4
    - C128 State pool (paged): size=F*R/swa_page_size*c128_ring_size, layers=num_layers_ca128
    - C4 Indexer State pool: size=F*R/swa_page_size*c4_ring_size, layers=num_layers_ca4

    Where F = full_token, R = swa_ratio
    Ring sizes: c4_ring_size=16 (or 8 for speculative), c128_ring_size=256 (or 128 for speculative)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        page_size: int,
        swa_ratio: float,
        is_speculative: bool = False,
    ):
        self.qk_nope_head_dim = model_config.qk_nope_head_dim
        self.qk_rope_head_dim = model_config.qk_rope_head_dim
        self.indexer_head_dim = model_config.index_head_dim
        self.compression_ratios = model_config.compress_ratios
        # NOTE: Hardcored swa page size from swa window size
        self.swa_page_size = model_config.window_size
        self.page_size = page_size
        self.swa_ratio = swa_ratio
        self.is_speculative = is_speculative

        # Get ring sizes based on speculative mode
        self.c4_ring_size = get_compress_state_ring_size(4, self.is_speculative)
        self.c128_ring_size = get_compress_state_ring_size(128, self.is_speculative)

        # Count layers by compression type
        self.num_layers_total = len(self.compression_ratios)
        self.num_layers_ca4 = sum(1 for r in self.compression_ratios if r == 4)
        self.num_layers_ca128 = sum(1 for r in self.compression_ratios if r == 128)

        # Bytes per full token
        self.bytes_per_full_token = self.get_bytes_per_full_token()

    def get_bytes_per_full_token(self) -> float:
        """Calculate total memory bytes per full_token.

        Returns:
            Total memory bytes per full_token (across all pools and layers)
        """
        # KV pool bytes per token (fp8 nope + bf16 rope + scale)
        # Layout: nope_fp8 (448) + rope_bf16 (64*2) + scale (8)
        kv_bytes = self.qk_nope_head_dim + self.qk_rope_head_dim * 2 + 8

        # Indexer bytes per token (fp8 + fp32 scale)
        # Layout: index_k (512) + scale (512/128*4)
        quant_block_size = 128
        indexer_bytes = (
            self.indexer_head_dim + self.indexer_head_dim // quant_block_size * 4
        )

        # State bytes per token (float32)
        # KVAndScore layout: (size, 2 * (1 + overlap) * head_dim)
        attn_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        state_dtype_size = 4  # float32
        c4_state_bytes = 2 * 2 * attn_head_dim * state_dtype_size  # overlap=True
        c128_state_bytes = 2 * 1 * attn_head_dim * state_dtype_size  # overlap=False
        c4_indexer_state_bytes = 2 * 2 * self.indexer_head_dim * state_dtype_size

        # State paged expansion: state size = num_swa_pages * ring_size
        # where num_swa_pages = swa_tokens / swa_page_size
        c4_state_ratio = self.c4_ring_size / self.swa_page_size
        c128_state_ratio = self.c128_ring_size / self.swa_page_size

        # Calculate total bytes per full_token
        bytes_per_full_token = (
            # SWA KV pool: size = full_token * swa_ratio
            self.swa_ratio * kv_bytes * self.num_layers_total
            # C4 KV pool: size = full_token / 4
            + 1 / 4 * kv_bytes * self.num_layers_ca4
            # C128 KV pool: size = full_token / 128
            + 1 / 128 * kv_bytes * self.num_layers_ca128
            # C4 indexer pool: size = full_token / 4
            + 1 / 4 * indexer_bytes * self.num_layers_ca4
            # C4 compress state pool (paged): size = num_swa_pages * c4_ring_size
            + self.swa_ratio * c4_state_ratio * c4_state_bytes * self.num_layers_ca4
            # C128 compress state pool (paged): size = num_swa_pages * c128_ring_size
            + self.swa_ratio
            * c128_state_ratio
            * c128_state_bytes
            * self.num_layers_ca128
            # C4 indexer compress state pool (paged): size = num_swa_pages * c4_ring_size
            + self.swa_ratio
            * c4_state_ratio
            * c4_indexer_state_bytes
            * self.num_layers_ca4
        )

        return bytes_per_full_token

    def calculate_pool_sizes(self, available_bytes: int) -> DSv4PoolSizes:
        """Calculate pool sizes based on available memory.

        Args:
            available_bytes: Available memory bytes for KV cache

        Returns:
            DSv4PoolSizes containing all pool sizes
        """
        full_token = int(available_bytes / self.bytes_per_full_token)

        # Align to page_size
        full_token = full_token // self.page_size * self.page_size

        # Calculate each pool's size
        swa_tokens = int(full_token * self.swa_ratio) // self.page_size * self.page_size

        pool_sizes = DSv4PoolSizes(
            full_max_total_num_tokens=full_token,
            swa_max_total_num_tokens=swa_tokens,
            c4_max_total_num_tokens=full_token // 4,
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
        # Profile available memory bytes directly (standalone path for DSv4)
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
    """Profile available memory bytes for KV cache.

    Args:
        device: Device type (cuda, etc.)
        gpu_id: GPU ID
        total_gpu_memory: Total GPU memory in GB
        mem_fraction_static: Static memory fraction
        distributed: Whether running in distributed mode
        cpu_group: CPU group for distributed

    Returns:
        Available memory bytes for KV cache
    """
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

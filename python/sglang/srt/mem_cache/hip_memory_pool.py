from __future__ import annotations

from typing import TYPE_CHECKING
import logging

import math
import torch
import triton
from hip import HiPAttentionOutputMetadata

if TYPE_CHECKING:
    from sglang.srt.layers.attention.hip_attention.hip_config import HiPAttentionConfig

logger = logging.getLogger(__name__)


class HiPMetadataCachePool:

    def __init__(
        self,
        size: int,  # max_total_num_tokens = bsz * seq_len
        head_num: int,
        layer_num: int,
        device: str,
        hip_config: HiPAttentionConfig,
    ):
        self.indices_pool = []
        self.ks_pool = []
        self.ks_count_pool = []
        self.ks_start_end_pool = []

        self.hip_dimens = []
        self.head_num = head_num

        metadata_pool_bytes = 0
        for layer_idx in range(layer_num):
            require_dense = (
                layer_idx in hip_config.dense_layers or
                hip_config.decode_always_dense or
                hip_config.force_dense
            )
            if len(hip_config.layers) == 2:
                layer_config = hip_config.layers[0 if require_dense else 1]
            else:
                layer_config = hip_config.layers[layer_idx]

            block_size_q = layer_config.stages[0].stage_block_size_q  # BLOCK_SIZE_Q
            stage_stride = layer_config.stages[0].stage_stride

            max_bdst_scan_times_bsz = max(
                math.ceil(math.ceil((size // bsz) / block_size_q) / stage_stride) * bsz
                for bsz in range(1, size // 16 + 1)  # FIXME: Assume 16 is the min sequence length for full batch
            )

            q_blocks = max(1, block_size_q // hip_config.block_sparse_block_size_q)
            n_chunks = triton.cdiv(layer_config.second_stage_k, layer_config.stages[-1].stage_chunk_size)

            self.hip_dimens.append((block_size_q, stage_stride, max_bdst_scan_times_bsz, q_blocks, n_chunks))

            self.indices_pool.append(
                torch.zeros((max_bdst_scan_times_bsz * head_num * q_blocks, n_chunks),
                            dtype=torch.int64, device=device)
            )
            metadata_pool_bytes += self.indices_pool[-1].numel() * self.indices_pool[-1].element_size()
            self.ks_pool.append(
                torch.zeros((max_bdst_scan_times_bsz * head_num * q_blocks, n_chunks),
                            dtype=torch.int32, device=device)
            )
            metadata_pool_bytes += self.ks_pool[-1].numel() * self.ks_pool[-1].element_size()
            self.ks_count_pool.append(
                torch.zeros((max_bdst_scan_times_bsz * head_num * q_blocks, 1),
                            dtype=torch.int32, device=device)
            )
            metadata_pool_bytes += self.ks_count_pool[-1].numel() * self.ks_count_pool[-1].element_size()
            self.ks_start_end_pool.append(
                torch.zeros((max_bdst_scan_times_bsz * head_num * q_blocks, 2),
                            dtype=torch.int32, device=device)
            )
            metadata_pool_bytes += self.ks_start_end_pool[-1].numel() * self.ks_start_end_pool[-1].element_size()

        logger.info(f"Allocated HiP metadata cache pool size: {metadata_pool_bytes / 1024 / 1024:.2f} MB")

    def get_hip_metadata_cache(self, layer_id: int, size: int, batch_size: int):
        block_size_q, stage_stride, max_bdst_scan_times_bsz, q_blocks, n_chunks = self.hip_dimens[layer_id]

        seq_len = size // batch_size

        bdst_scan = triton.cdiv(triton.cdiv(seq_len, block_size_q), stage_stride)
        first_dim = batch_size * self.head_num * bdst_scan * q_blocks

        indices = self.indices_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, n_chunks)
        ks = self.ks_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, n_chunks)
        ks_count = self.ks_count_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, 1)
        ks_start_end = self.ks_start_end_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, 2)

        return HiPAttentionOutputMetadata(
            indices=indices,
            ks=ks,
            ks_count=ks_count,
            ks_start_end=ks_start_end,
            key_access_log=None,
            key_access_count=None,
            block_access_log=None,
            block_access_score=None,
            block_access_count=None,
        )

    def set_hip_metadata_cache(
        self,
        layer_id: int,
        size: int,
        batch_size: int,
        cache_loc: torch.Tensor,  # FIXME: do something with cache_loc
        metadata: HiPAttentionOutputMetadata
    ):
        block_size_q, stage_stride, max_bdst_scan_times_bsz, q_blocks, n_chunks = self.hip_dimens[layer_id]

        seq_len = size // batch_size

        bdst_scan = triton.cdiv(triton.cdiv(seq_len, block_size_q), stage_stride)
        first_dim = batch_size * self.head_num * bdst_scan * q_blocks

        self.indices_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, n_chunks) \
            .copy_(metadata.indices)
        self.ks_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, n_chunks) \
            .copy_(metadata.ks)
        self.ks_count_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, 1) \
            .copy_(metadata.ks_count)
        self.ks_start_end_pool[layer_id][:first_dim] \
            .view(batch_size * self.head_num, bdst_scan * q_blocks, 2) \
            .copy_(metadata.ks_start_end)

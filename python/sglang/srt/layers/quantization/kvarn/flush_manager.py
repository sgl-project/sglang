# SPDX-License-Identifier: Apache-2.0
"""KVarN tile flush manager.

Manages the lifecycle of KV cache blocks in the dual-pool architecture:
  - **Tail pool** (fp16): stores rotated K/V for in-progress + sink blocks.
    Each block occupies one tail-pool slot: ``[pool_slots, group, Hk, D]``.
  - **Compressed cache** (int4/uint8): stores flushed blocks as packed tiles.
    ``[num_blocks, Hk, tile_bytes]`` per layer.

Flush = compress fp16 tail pool → int4 cache, then free the tail pool slot.
Dequant = read int4 cache → fp16 (on demand, for attention).
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.dequant import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from sglang.srt.layers.quantization.kvarn.sinkhorn import variance_normalize_batched
from sglang.srt.layers.quantization.kvarn.store import (
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v_batch_from_sinkhorn,
)

logger = logging.getLogger(__name__)


class KVarNFlushManager:
    """Manages the fp16 tail pool ↔ int4 compressed cache lifecycle."""

    def __init__(
        self,
        cfg: KVarNConfig,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        sink_blocks: int = 1,
    ):
        self.cfg = cfg
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim else head_dim
        self.group = cfg.group
        self.tile_bytes = cfg.tile_bytes_aligned
        self.sink_blocks = sink_blocks

    def flush_block(
        self,
        block_id: int,
        tail_K: list[torch.Tensor],  # per-layer [pool_slots, group, Hk, D] fp16
        tail_V: list[torch.Tensor],  # per-layer [pool_slots, group, Hk, vD] fp16
        slot: int,
        compressed_cache: list[
            torch.Tensor
        ],  # per-layer [num_blocks, Hk, tile_bytes] uint8
    ):
        """Compress one block from the tail pool to the int4 cache.

        Reads K/V from tail_K[layer][slot] / tail_V[layer][slot] (shape
        ``[group, Hk, D]``), applies Sinkhorn + RTN, and writes packed tiles
        to compressed_cache[layer][block_id].
        """
        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        vD = self.v_head_dim
        G = self.group

        for layer_id in range(self.num_layers):
            K_blk = tail_K[layer_id][slot]  # [G, Hk, D] fp16 (rotated)
            V_blk = tail_V[layer_id][slot]  # [G, Hk, vD] fp16 (rotated)

            # K tile orientation: [D, group] per head (channels × tokens)
            # K_blk is [group, Hk, D] → per-head [group, D] → transpose to [D, group]
            K_per_head = K_blk.permute(1, 2, 0)  # [Hk, D, group]
            # V tile orientation: [group, D] per head (tokens × channels)
            V_per_head = V_blk.permute(1, 0, 2)  # [Hk, group, vD]

            # Sinkhorn variance normalization (batched over heads)
            K_bal, K_sc, K_sr = variance_normalize_batched(
                K_per_head, iterations=cfg.sinkhorn_iters
            )
            V_bal, V_sc, V_sr = variance_normalize_batched(
                V_per_head, iterations=cfg.sinkhorn_iters
            )

            # RTN quantization + scale absorption + packing
            K_out = kvarn_store_tile_k_batch_from_sinkhorn(
                K_bal,
                K_sc.squeeze(1),  # [Hk, group]
                K_sr.squeeze(2),  # [Hk, D]
                bits=cfg.key_bits,
            )
            V_out = kvarn_store_tile_v_batch_from_sinkhorn(
                V_bal,
                V_sc.squeeze(1),  # [Hk, vD]
                V_sr.squeeze(2),  # [Hk, group]
                bits=cfg.value_bits,
            )

            # Write packed tiles to compressed cache
            cache = compressed_cache[layer_id]
            for h in range(Hk):
                self._write_packed_tile_k(
                    cache,
                    block_id,
                    h,
                    K_out["q_packed_uint8"][h],
                    K_out["s_col_K"][h],
                    K_out["zp_K"][h],
                    K_out["s_row_K"][h],
                )
                self._write_packed_tile_v(
                    cache,
                    block_id,
                    h,
                    V_out["q_packed_uint8"][h],
                    V_out["s_col_V"][h],
                    V_out["s_row_V"][h],
                    V_out["zp_V"][h],
                )

    def flush_blocks_batched(
        self,
        block_ids: list[int],
        tail_K: list[torch.Tensor],
        tail_V: list[torch.Tensor],
        slots: list[int],
        compressed_cache: list[torch.Tensor],
    ):
        """Compress multiple blocks in a batched fashion for efficiency."""
        if not block_ids:
            return

        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        vD = self.v_head_dim
        G = self.group
        nB = len(block_ids)

        for layer_id in range(self.num_layers):
            # Gather all blocks' K/V data
            K_blocks = torch.stack([tail_K[layer_id][slots[i]] for i in range(nB)])
            # K_blocks: [nB, G, Hk, D] → permute to [nB*Hk, D, G] for K path
            K_per_head = K_blocks.permute(0, 2, 3, 1)  # [nB, Hk, D, G]
            K_tiles = K_per_head.reshape(nB * Hk, D, G)

            V_blocks = torch.stack([tail_V[layer_id][slots[i]] for i in range(nB)])
            # V_blocks: [nB, G, Hk, vD] → permute to [nB*Hk, G, vD] for V path
            V_per_head = V_blocks.permute(0, 2, 1, 3)  # [nB, Hk, G, vD]
            V_tiles = V_per_head.reshape(nB * Hk, G, vD)

            # Sinkhorn + RTN
            K_bal, K_sc, K_sr = variance_normalize_batched(
                K_tiles, iterations=cfg.sinkhorn_iters
            )
            V_bal, V_sc, V_sr = variance_normalize_batched(
                V_tiles, iterations=cfg.sinkhorn_iters
            )

            K_out = kvarn_store_tile_k_batch_from_sinkhorn(
                K_bal,
                K_sc.squeeze(1),  # [nB*Hk, G]
                K_sr.squeeze(2),  # [nB*Hk, D]
                bits=cfg.key_bits,
            )
            V_out = kvarn_store_tile_v_batch_from_sinkhorn(
                V_bal,
                V_sc.squeeze(1),  # [nB*Hk, vD]
                V_sr.squeeze(2),  # [nB*Hk, G]
                bits=cfg.value_bits,
            )

            cache = compressed_cache[layer_id]
            for i, bid in enumerate(block_ids):
                for h in range(Hk):
                    idx = i * Hk + h
                    self._write_packed_tile_k(
                        cache,
                        bid,
                        h,
                        K_out["q_packed_uint8"][idx],
                        K_out["s_col_K"][idx],
                        K_out["zp_K"][idx],
                        K_out["s_row_K"][idx],
                    )
                    self._write_packed_tile_v(
                        cache,
                        bid,
                        h,
                        V_out["q_packed_uint8"][idx],
                        V_out["s_col_V"][idx],
                        V_out["s_row_V"][idx],
                        V_out["zp_V"][idx],
                    )

        logger.info(f"KVarN flush: {nB} blocks compressed to int4")

    def flush_batched_fast(
        self,
        block_ids: list[int],
        tail_K: list[torch.Tensor],
        tail_V: list[torch.Tensor],
        slots: list[int],
        compressed_cache: list[torch.Tensor],
    ):
        """Batched flush with vectorized write — one index_select gather +
        one index_copy scatter per (layer, chunk), replacing the per-(block,head)
        Python write loop. Numerically identical to flush_blocks_batched.

        Assembles packed cache records via
        tensor concatenation in config-offset order and writes with a single
        scatter.
        """
        if not block_ids:
            return

        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        vD = self.v_head_dim
        G = self.group
        T = cfg.tile_bytes_aligned
        kpb = cfg.k_packed_bytes
        vpb = cfg.v_packed_bytes
        nB = len(block_ids)

        # Defensive bounds check: block_ids and slots must be in range.
        # A failure here indicates the compressed cache or tail pool is
        # undersized relative to the scheduler's page allocator.
        max_bid = max(block_ids)
        max_slot = max(slots)
        if max_bid >= compressed_cache[0].shape[0]:
            raise IndexError(
                f"KVarN flush: block_id {max_bid} >= compressed_blocks "
                f"{compressed_cache[0].shape[0]}"
            )
        if max_slot >= tail_K[0].shape[0]:
            raise IndexError(
                f"KVarN flush: slot {max_slot} >= tail_pool_slots "
                f"{tail_K[0].shape[0]}"
            )

        # Chunk to bound transient gather memory
        CHUNK_BLOCKS = max(1, 2048 // max(Hk, 1))

        slots_dev = torch.as_tensor(slots, dtype=torch.long, device=tail_K[0].device)
        bids_dev = torch.as_tensor(block_ids, dtype=torch.long, device=tail_K[0].device)

        for layer_id in range(self.num_layers):
            kvc = compressed_cache[layer_id]
            for c0 in range(0, nB, CHUNK_BLOCKS):
                bchunk = block_ids[c0 : c0 + CHUNK_BLOCKS]
                slot_t = slots_dev[c0 : c0 + CHUNK_BLOCKS]
                bid_t = bids_dev[c0 : c0 + CHUNK_BLOCKS]
                nBc = len(bchunk)

                # One gather per chunk
                K_rot = (
                    tail_K[layer_id].index_select(0, slot_t).float()
                )  # [nBc, G, Hk, D]
                V_rot = tail_V[layer_id].index_select(0, slot_t).float()

                # Tiles: K [N, D, G] (absorb=channel), V [N, G, D] (absorb=token)
                K_tiles = K_rot.permute(0, 2, 3, 1).reshape(nBc * Hk, D, G)
                V_tiles = V_rot.permute(0, 2, 1, 3).reshape(nBc * Hk, G, vD)

                # Sinkhorn + RTN (fused when square, else separate)
                if K_tiles.shape[1:] == V_tiles.shape[1:]:
                    # Square: fuse into one Sinkhorn launch
                    K_bal, K_sc, K_sr = variance_normalize_batched(
                        torch.cat([K_tiles, V_tiles], dim=0),
                        iterations=cfg.sinkhorn_iters,
                    )
                    nk = nBc * Hk
                    K_out = kvarn_store_tile_k_batch_from_sinkhorn(
                        K_bal[:nk], K_sc[:nk], K_sr[:nk], bits=cfg.key_bits
                    )
                    V_out = kvarn_store_tile_v_batch_from_sinkhorn(
                        K_bal[nk:], K_sc[nk:], K_sr[nk:], bits=cfg.value_bits
                    )
                else:
                    # Non-square (e.g. head_dim=256, group=128): separate launches
                    K_bal, K_sc, K_sr = variance_normalize_batched(
                        K_tiles, iterations=cfg.sinkhorn_iters
                    )
                    V_bal, V_sc, V_sr = variance_normalize_batched(
                        V_tiles, iterations=cfg.sinkhorn_iters
                    )
                    K_out = kvarn_store_tile_k_batch_from_sinkhorn(
                        K_bal, K_sc.squeeze(1), K_sr.squeeze(2), bits=cfg.key_bits
                    )
                    V_out = kvarn_store_tile_v_batch_from_sinkhorn(
                        V_bal, V_sc.squeeze(1), V_sr.squeeze(2), bits=cfg.value_bits
                    )

                # Assemble packed cache record [nBc*Hk, tile_bytes] via concatenation
                M = nBc * Hk
                parts = [
                    K_out["q_packed_uint8"].reshape(M, kpb),
                    K_out["s_col_K"].contiguous().view(torch.uint8),
                    K_out["zp_K"].contiguous().view(torch.uint8),
                    K_out["s_row_K"].contiguous().view(torch.uint8),
                    V_out["q_packed_uint8"].reshape(M, vpb),
                    V_out["s_col_V"].contiguous().view(torch.uint8),
                    V_out["s_row_V"].contiguous().view(torch.uint8),
                    V_out["zp_V"].contiguous().view(torch.uint8),
                ]
                rec = torch.cat(parts, dim=1)  # [M, tile_bytes]
                if rec.shape[1] < T:
                    rec = torch.nn.functional.pad(rec, (0, T - rec.shape[1]))

                # One scatter per chunk
                kvc[bid_t] = rec.view(nBc, Hk, T)

    def dequant_block(
        self,
        block_id: int,
        compressed_cache: list[torch.Tensor],
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a block from int4 cache and dequantize.

        Returns (K [group, Hk, D] fp16, V [group, Hk, vD] fp16) in the
        ROTATED frame (caller must apply H^-1 if un-rotated data is needed).
        """
        cfg = self.cfg
        Hk = self.num_kv_heads
        D = self.head_dim
        vD = self.v_head_dim
        G = self.group
        pack_k = 8 // cfg.key_bits
        pack_v = 8 // cfg.value_bits

        cache = compressed_cache[layer_id][block_id]  # [Hk, tile_bytes] uint8

        K_out = torch.zeros(G, Hk, D, dtype=torch.float16, device=cache.device)
        V_out = torch.zeros(G, Hk, vD, dtype=torch.float16, device=cache.device)

        for h in range(Hk):
            # Dequant K tile: packed as [D, G // pack_k]
            off = cfg.k_packed_offset
            k_packed = cache[h, off : off + D * (G // pack_k)].reshape(D, G // pack_k)
            off = cfg.k_s_col_offset
            s_col_K = cache[h, off : off + D * 2].view(torch.float16)
            off = cfg.k_zp_offset
            zp_K = cache[h, off : off + D * 2].view(torch.float16)
            off = cfg.k_s_row_offset
            s_row_K = cache[h, off : off + G * 2].view(torch.float16)

            K_deq = kvarn_dequant_tile_k(
                k_packed,
                s_col_K,
                zp_K,
                s_row_K,
                group=G,
                bits=cfg.key_bits,
            )
            # K_deq is [D, G] in rotated frame → transpose to [G, D]
            K_out[:, h, :] = K_deq.t().to(torch.float16)

            # Dequant V tile: packed as [G, D // pack_v]
            off = cfg.v_packed_offset
            v_packed = cache[h, off : off + G * (vD // pack_v)].reshape(G, vD // pack_v)
            off = cfg.v_s_col_offset
            s_col_V = cache[h, off : off + vD * 2].view(torch.float16)
            off = cfg.v_s_row_offset
            s_row_V = cache[h, off : off + G * 2].view(torch.float16)
            off = cfg.v_zp_offset
            zp_V = cache[h, off : off + G * 2].view(torch.float16)

            V_deq = kvarn_dequant_tile_v(
                v_packed,
                s_col_V,
                s_row_V,
                zp_V,
                head_dim=vD,
                bits=cfg.value_bits,
            )
            # V_deq is [G, vD] in rotated frame
            V_out[:, h, :] = V_deq.to(torch.float16)

        return K_out, V_out

    def _write_packed_tile_k(
        self,
        cache: torch.Tensor,
        block_id: int,
        head_id: int,
        q_packed: torch.Tensor,
        s_col: torch.Tensor,
        zp: torch.Tensor,
        s_row: torch.Tensor,
    ):
        """Write one packed K tile to the compressed cache."""
        cfg = self.cfg
        off = cfg.k_packed_offset
        cache[block_id, head_id, off : off + q_packed.numel()].copy_(q_packed.flatten())
        off = cfg.k_s_col_offset
        cache[block_id, head_id, off : off + s_col.numel() * 2].copy_(
            s_col.view(torch.uint8).flatten()
        )
        off = cfg.k_zp_offset
        cache[block_id, head_id, off : off + zp.numel() * 2].copy_(
            zp.view(torch.uint8).flatten()
        )
        off = cfg.k_s_row_offset
        cache[block_id, head_id, off : off + s_row.numel() * 2].copy_(
            s_row.view(torch.uint8).flatten()
        )

    def _write_packed_tile_v(
        self,
        cache: torch.Tensor,
        block_id: int,
        head_id: int,
        q_packed: torch.Tensor,
        s_col: torch.Tensor,
        s_row: torch.Tensor,
        zp: torch.Tensor,
    ):
        """Write one packed V tile to the compressed cache."""
        cfg = self.cfg
        off = cfg.v_packed_offset
        cache[block_id, head_id, off : off + q_packed.numel()].copy_(q_packed.flatten())
        off = cfg.v_s_col_offset
        cache[block_id, head_id, off : off + s_col.numel() * 2].copy_(
            s_col.view(torch.uint8).flatten()
        )
        off = cfg.v_s_row_offset
        cache[block_id, head_id, off : off + s_row.numel() * 2].copy_(
            s_row.view(torch.uint8).flatten()
        )
        off = cfg.v_zp_offset
        cache[block_id, head_id, off : off + zp.numel() * 2].copy_(
            zp.view(torch.uint8).flatten()
        )

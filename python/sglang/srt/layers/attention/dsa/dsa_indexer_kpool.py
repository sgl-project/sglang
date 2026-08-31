from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import PretrainedConfig

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    DUAL_STREAM_TOKEN_THRESHOLD,
    BaseIndexerMetadata,
    rotate_activation,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import add_prefix, ceil_align, is_cuda, is_hip, is_npu

if is_cuda():
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if is_npu():
    import custom_ops  # noqa: F401

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.dsa.utils import (
    aiter_can_use_preshuffle_paged_mqa,
    cp_zigzag_full_plan_rows,
    dsa_use_prefill_cp,
    is_dsa_enable_prefill_cp,
    is_dsa_prefill_cp_in_seq_split,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.runtime_context import get_device, get_parallel

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool


class IndexerKPool(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        config: Optional[PretrainedConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.compress_gate_stream = None
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.cp_size = get_parallel().attn_cp_size if self.dsa_enable_prefill_cp else 1
        self.skip_rope = skip_rope

        self.index_kpool = config.index_kpool
        self.index_kpool_always_select_tail = config.index_kpool_always_select_tail
        self.index_kpool_compress = config.index_kpool_compress

        assert (
            self.index_kpool > 1
            and self.index_kpool_compress
            and self.index_kpool_always_select_tail
        )

        assert self.index_topk % self.index_kpool == 0, (
            f"index_topk ({self.index_topk}) must be divisible by "
            f"index_kpool ({self.index_kpool})"
        )
        assert (
            64 % self.index_kpool == 0
        ), f"index_kpool ({self.index_kpool}) must divide page_size (64)"

        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.index_kpool, self.head_dim, dtype=torch.float32)
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(self.head_dim, self.hidden_size, dtype=torch.bfloat16)
        )

        if is_cuda() and self.alt_stream is not None:
            self.compress_gate_stream = torch.cuda.Stream()

        if is_cuda():
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        # Keep weights_proj in FP32 because its accumulation/scale path requires FP32;
        # checkpoint BF16 weights are cast on load.
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.float32,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.k_norm = LayerNorm(self.head_dim, dtype=torch.float32)
        if not self.skip_rope:
            self.rotary_emb = get_rope_wrapper(
                rope_head_dim,
                rotary_dim=rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,  # type: ignore
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                device=get_device().device,
            )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5

    @torch.compile(dynamic=True)
    def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
        weights, _ = self.weights_proj(x.float())
        weights = weights * self.n_heads**-0.5
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        return weights

    @staticmethod
    def _fp8_mqa_logits(
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        *,
        clean_logits: bool,
    ) -> torch.Tensor:
        if is_hip():
            from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

            return fp8_mqa_logits(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                starts,
                ends,
                clean_logits=clean_logits,
            )

        return deep_gemm.fp8_mqa_logits(
            q_fp8,
            (k_fp8, k_scale),
            weights,
            starts,
            ends,
            clean_logits=clean_logits,
        )

    @staticmethod
    def _cp_gather_concat(
        tensors: List[torch.Tensor],
        cp_size: int,
        forward_batch: ForwardBatch,
    ) -> List[torch.Tensor]:
        if not tensors:
            return []
        n_local = tensors[0].shape[0]
        flats = []
        feature_sizes = []
        tails = []
        for tensor in tensors:
            assert tensor.shape[0] == n_local
            assert tensor.dtype == tensors[0].dtype
            tails.append(tensor.shape[1:])
            flat = tensor.reshape(n_local, -1).contiguous()
            feature_sizes.append(flat.shape[1])
            flats.append(flat)
        combined = flats[0] if len(flats) == 1 else torch.cat(flats, dim=1)
        gathered = cp_all_gather_rerange_output(
            combined,
            cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
        parts = torch.split(gathered, feature_sizes, dim=1)
        return [part.reshape(part.shape[0], *tail) for part, tail in zip(parts, tails)]

    @staticmethod
    def _get_index_k_read_buffer(pool, layer_id: int) -> torch.Tensor:
        if hasattr(pool, "get_broadcastable_index_k_with_scale_buffer"):
            return pool.get_broadcastable_index_k_with_scale_buffer(layer_id)
        if hasattr(pool, "_get_broadcastable_index_buffer"):
            return pool._get_broadcastable_index_buffer(layer_id)
        return pool.get_index_k_with_scale_buffer(layer_id=layer_id)

    def _write_compressed_pooled_index_cache(
        self,
        slot_k,
        slot_score,
        write_locs,
        forward_batch,
        layer_id,
        write_mask=None,
        return_compressed: bool = False,
        write_cache: bool = True,
    ):
        if slot_k.shape[0] == 0:
            if return_compressed:
                return (
                    torch.empty(
                        (0, self.head_dim),
                        dtype=torch.float8_e4m3fn,
                        device=slot_k.device,
                    ),
                    torch.empty((0,), dtype=torch.float32, device=slot_k.device),
                )
            return None
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            kpool_softmax_rotate_write_cache,
        )

        pool = get_token_to_kv_pool()
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            if not return_compressed:
                return None
            write_cache = False

        buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
        return kpool_softmax_rotate_write_cache(
            pool=pool,
            buf=buf,
            slot_k=slot_k,
            slot_score=slot_score,
            ape=self.index_kpool_compress_ape,
            loc=write_locs.contiguous(),
            write_mask=write_mask.contiguous() if write_mask is not None else None,
            round_scale=self.scale_fmt is not None,
            return_compressed=return_compressed,
            write_cache=write_cache,
        )

    @staticmethod
    def _write_returned_compressed_pooled_index_cache(
        forward_batch, layer_id, write_locs, k_fp8, k_scale
    ):
        if k_fp8.shape[0] == 0:
            return
        get_token_to_kv_pool().set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=write_locs.contiguous(),
            index_k=k_fp8.contiguous(),
            index_k_scale=k_scale.contiguous(),
        )

    def _compress_write_decode(
        self,
        key,
        gate_score,
        positions,
        forward_batch,
        layer_id,
        metadata,
    ):
        batch = key.shape[0]
        if batch == 0:
            return

        pool = get_token_to_kv_pool()
        if hasattr(pool, "invalidate_index_buffer_for_layer"):
            pool.invalidate_index_buffer_for_layer(layer_id)
        if hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id):
            return

        pool.kpool_decode_update_index_cache(
            layer_id=layer_id,
            key=key,
            slot_score=gate_score,
            ape=self.index_kpool_compress_ape,
            block_tables=metadata.get_page_table_64(),
            req_pool_indices=forward_batch.req_pool_indices[:batch],
            positions=positions[:batch],
            seq_lens=metadata.get_seqlens_int32()[:batch],
            out_cache_loc=forward_batch.out_cache_loc[:batch],
            round_scale=self.scale_fmt is not None,
        )

    def _compress_write_extend(
        self,
        key,
        gate_score,
        positions,
        forward_batch,
        layer_id,
        metadata,
        return_compressed: bool = False,
        write_cache: bool = True,
    ):
        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        attn_metadata = metadata.attn_metadata
        plan = attn_metadata.kpool_extend_plan
        if plan is not None:
            import os

            from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
                all_gather_and_scatter_pool_slots,
                kpool_assemble_softmax_rotate_write_cache,
                scatter_kpool_tail_updates,
            )

            pool = get_token_to_kv_pool()
            if hasattr(pool, "invalidate_index_buffer_for_layer"):
                pool.invalidate_index_buffer_for_layer(layer_id)
            layer_owned = not (
                hasattr(pool, "_is_layer_owned") and not pool._is_layer_owned(layer_id)
            )
            if not layer_owned:
                return None

            writes, tails, cp = plan.writes, plan.tails, plan.cp
            if writes.is_empty and tails.is_empty:
                return None

            tail_k_buf, tail_score_buf = pool.get_compress_tail_buffers(layer_id)
            layer_sharded = bool(getattr(pool, "layer_shard_enabled", False))
            cache_cp = None if layer_sharded else cp
            if os.environ.get("SGLANG_DSA_KPOOL_DEBUG_BOUNDS") == "1":
                if not writes.is_empty:
                    active = (
                        cache_cp.local_write_mask
                        if cache_cp is not None
                        else torch.ones_like(writes.req, dtype=torch.bool)
                    )
                    if bool(active.any().item()):
                        active_req = writes.req[active]
                        active_write_loc = writes.write_loc[active]
                        active_chunk_max = writes.chunk_src[active].to(
                            torch.long
                        ) + torch.clamp(
                            self.index_kpool
                            - writes.n_from_tail[active].to(torch.long)
                            - 1,
                            min=0,
                        )
                        max_chunk = int(active_chunk_max.max().item())
                        min_chunk = int(writes.chunk_src[active].min().item())
                        min_req = int(active_req.min().item())
                        max_req = int(active_req.max().item())
                        min_loc = int(active_write_loc.min().item())
                        max_loc = int(active_write_loc.max().item())
                        max_page = max_loc // int(pool.slots_per_page)
                        if (
                            min_chunk < 0
                            or max_chunk >= key.shape[0]
                            or max_chunk >= gate_score.shape[0]
                            or min_req < 0
                            or max_req >= tail_k_buf.shape[0]
                            or min_loc < 0
                            or max_page
                            >= pool.get_index_k_with_scale_buffer(
                                layer_id=layer_id
                            ).shape[0]
                        ):
                            raise RuntimeError(
                                "DSA kpool write plan out of bounds: "
                                f"{key.shape=}, {gate_score.shape=}, {tail_k_buf.shape=}, "
                                f"{pool.get_index_k_with_scale_buffer(layer_id=layer_id).shape=}, "
                                f"{min_chunk=}, {max_chunk=}, {min_req=}, {max_req=}, "
                                f"{min_loc=}, {max_loc=}, {pool.slots_per_page=}, "
                                f"cp_rank={cache_cp.rank if cache_cp is not None else None}, "
                                f"cp_size={cache_cp.size if cache_cp is not None else None}"
                            )
                if not tails.is_empty:
                    tail_chunk_max = tails.chunk_src.to(torch.long) + torch.clamp(
                        tails.n_write.to(torch.long) - 1, min=0
                    )
                    max_tail_chunk = int(tail_chunk_max.max().item())
                    min_tail_chunk = int(tails.chunk_src.min().item())
                    min_tail_req = int(tails.req.min().item())
                    max_tail_req = int(tails.req.max().item())
                    if (
                        min_tail_chunk < 0
                        or max_tail_chunk >= key.shape[0]
                        or min_tail_req < 0
                        or max_tail_req >= tail_k_buf.shape[0]
                    ):
                        raise RuntimeError(
                            "DSA kpool tail plan out of bounds: "
                            f"{key.shape=}, {tail_k_buf.shape=}, "
                            f"{min_tail_chunk=}, {max_tail_chunk=}, "
                            f"{min_tail_req=}, {max_tail_req=}"
                        )
            if not writes.is_empty:
                buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)
                kpool_assemble_softmax_rotate_write_cache(
                    pool=pool,
                    buf=buf,
                    chunk_k=key,
                    chunk_score=gate_score,
                    tail_k=tail_k_buf,
                    tail_score=tail_score_buf,
                    req_pool_idx=writes.req,
                    n_from_tail=writes.n_from_tail,
                    chunk_src_start=writes.chunk_src,
                    tail_logical_base=writes.tail_logical_base,
                    ape=self.index_kpool_compress_ape,
                    loc=writes.write_loc,
                    write_mask=(
                        cache_cp.local_write_mask if cache_cp is not None else None
                    ),
                    round_scale=self.scale_fmt is not None,
                )
                if cache_cp is not None and write_cache:
                    all_gather_and_scatter_pool_slots(
                        buf=buf,
                        local_locs=writes.write_loc,
                        owner_rank=cache_cp.owner_rank,
                        cp_size=cache_cp.size,
                        cp_rank=cache_cp.rank,
                        slots_per_page=pool.slots_per_page,
                    )

            if not tails.is_empty:
                scatter_kpool_tail_updates(
                    pool=pool,
                    chunk_k=key,
                    chunk_score=gate_score,
                    tail_k=tail_k_buf,
                    tail_score=tail_score_buf,
                    req_pool_idx=tails.req,
                    dst_logical_start=tails.dst_logical_start,
                    chunk_src_start=tails.chunk_src,
                    n_write=tails.n_write,
                )
            return None

        kpool = self.index_kpool
        block_tables = metadata.get_page_table_64()
        kpool_write_locs = getattr(attn_metadata, "kpool_extend_write_locs", None)
        q_offset = 0
        compressed_by_batch = (
            [None for _ in range(forward_batch.batch_size)]
            if return_compressed
            else None
        )

        for i in range(forward_batch.batch_size):
            q_len = int(forward_batch.extend_seq_lens_cpu[i])
            if q_len == 0:
                continue

            req_pool_idx = forward_batch.req_pool_indices[i].to(torch.long)
            key_chunk = key[q_offset : q_offset + q_len]
            score_chunk = gate_score[q_offset : q_offset + q_len]
            seq_len = int(forward_batch.seq_lens_cpu[i].item())
            first_pos = seq_len - q_len
            first_slot = first_pos % kpool

            if first_slot != 0:
                raise NotImplementedError(
                    "index_kpool_compress extend requires kpool-aligned chunk "
                    "starts. Set chunked_prefill_size % index_kpool == 0 and "
                    "avoid non-aligned prefix reuse."
                )

            pool_start_id = first_pos // kpool
            page_size = get_token_to_kv_pool().page_size
            use_returned_compressed = return_compressed and (
                not write_cache or pool_start_id == 0
            )
            n_pools = q_len // kpool
            n_drain = n_pools * kpool
            if n_pools > 0:
                slot_k = key_chunk[:n_drain].view(n_pools, kpool, self.head_dim)
                slot_score = score_chunk[:n_drain].view(n_pools, kpool, self.head_dim)
                write_locs = (
                    kpool_write_locs[i] if kpool_write_locs is not None else None
                )
                if write_locs is None:
                    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
                        compute_pooled_write_locs,
                    )

                    num_token_pages = (seq_len + page_size - 1) // page_size
                    token_page_table = block_tables[i, :num_token_pages].contiguous()
                    pool_ids = pool_start_id + torch.arange(
                        n_pools, dtype=torch.int64, device=key.device
                    )
                    write_locs = compute_pooled_write_locs(
                        token_page_table, pool_ids, kpool
                    )
                    compressed = self._write_compressed_pooled_index_cache(
                        slot_k,
                        slot_score,
                        write_locs,
                        forward_batch,
                        layer_id,
                        return_compressed=use_returned_compressed,
                        write_cache=write_cache,
                    )
                else:
                    compressed = self._write_compressed_pooled_index_cache(
                        slot_k,
                        slot_score,
                        write_locs,
                        forward_batch,
                        layer_id,
                        return_compressed=use_returned_compressed,
                        write_cache=write_cache,
                    )
                if compressed_by_batch is not None and compressed is not None:
                    compressed_by_batch[i] = (
                        pool_start_id,
                        compressed[0],
                        compressed[1],
                        write_locs.contiguous(),
                    )

            n_remain = q_len - n_drain
            get_token_to_kv_pool().set_compress_tail_for_request(
                layer_id=layer_id,
                req_pool_idx=req_pool_idx,
                key_tail=key_chunk[n_drain:] if n_remain > 0 else key_chunk[:0],
                score_tail=score_chunk[n_drain:] if n_remain > 0 else score_chunk[:0],
                n_remain=n_remain,
                dst_logical_start=first_pos + n_drain,
            )
            q_offset += q_len

        return compressed_by_batch

    def _compress_write(
        self,
        x,
        key,
        positions,
        forward_batch,
        layer_id,
        metadata,
        gate_score: Optional[torch.Tensor] = None,
        return_compressed: bool = False,
        write_cache: bool = True,
    ):
        if key.shape[0] == 0:
            return None

        if gate_score is None:
            gate_score = F.linear(x, self.index_kpool_compress_gate)

        if forward_batch.forward_mode.is_decode_or_idle():
            self._compress_write_decode(
                key=key,
                gate_score=gate_score,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
            )
        elif forward_batch.forward_mode.is_extend():
            return self._compress_write_extend(
                key=key,
                gate_score=gate_score,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
                return_compressed=return_compressed,
                write_cache=write_cache,
            )
        else:
            raise NotImplementedError(
                "index_kpool_compress currently supports decode and extend only."
            )
        return None

    def _compute_gate_score_if_missing(
        self, x: torch.Tensor, gate_score: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if gate_score is not None:
            return gate_score
        return F.linear(x, self.index_kpool_compress_gate)

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        precompute_compress_gate: bool = False,
    ):
        gate_score = None
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if precompute_compress_gate:
                assert self.compress_gate_stream is not None
                self.compress_gate_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            if precompute_compress_gate:
                with torch.cuda.stream(self.compress_gate_stream):
                    gate_score = F.linear(x, self.index_kpool_compress_gate)

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        if not self.skip_rope:
            q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

            query[..., : self.rope_head_dim] = q_rope
            key[..., : self.rope_head_dim] = k_rope

        query = rotate_activation(query)

        return query, key, gate_score

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ):
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        if not self.skip_rope:
            _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
            key[..., : self.rope_head_dim] = k_rope
        return key

    def _full_topk_for_short_sequence(
        self, metadata: BaseIndexerMetadata, device: torch.device
    ) -> torch.Tensor:
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=device,
        )
        topk_full = metadata.topk_transform(dummy_logits, self.index_topk)
        if self.index_kpool == 1:
            return topk_full
        padding = torch.full(
            (topk_full.shape[0], self.index_kpool - 1),
            -1,
            dtype=topk_full.dtype,
            device=topk_full.device,
        )
        return torch.cat([topk_full, padding], dim=1)

    def _topk_from_kpool_logits(
        self,
        logits: torch.Tensor,
        pool_lens: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        topk_offsets: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        out_rows: Optional[int] = None,
        page_table_row_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            topk_from_pooled_history_logits,
        )

        n_rows = logits.shape[0]
        if (
            page_table is not None
            and page_table_row_index is None
            and page_table.shape[0] != n_rows
        ):
            page_table = page_table[:n_rows]
        if topk_offsets is not None and topk_offsets.shape[0] != n_rows:
            topk_offsets = topk_offsets[:n_rows]
        if page_table_row_index is not None and page_table_row_index.shape[0] != n_rows:
            page_table_row_index = page_table_row_index[:n_rows]

        return topk_from_pooled_history_logits(
            logits=logits,
            group_lengths=pool_lens,
            pool_size=self.index_kpool,
            topk=self.index_topk,
            page_table=page_table,
            topk_offsets=topk_offsets,
            seq_lens=seq_lens,
            row_starts=row_starts,
            out_rows=out_rows,
            page_table_row_index=page_table_row_index,
        )

    def _get_kpool_decode_metadata(
        self,
        metadata: BaseIndexerMetadata,
        block_tables: torch.Tensor,
        seqlens_32: torch.Tensor,
        blocksize: int,
        build_schedule_metadata: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            build_pooled_page_table_64,
        )

        attn_metadata = metadata.attn_metadata
        plan = attn_metadata.kpool_write_plan
        if plan is not None and plan.pool_seqlens_per_q is not None:
            pool_seqlens = plan.pool_seqlens_per_q[: seqlens_32.shape[0]]
            pool_context_lens = pool_seqlens.contiguous().view(-1, 1)
            pool_block_tables = build_pooled_page_table_64(
                block_tables, self.index_kpool
            ).contiguous()
            pool_schedule_metadata = plan.pool_schedule_metadata
            if pool_schedule_metadata is None and build_schedule_metadata:
                pool_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    pool_context_lens.clamp(min=1), blocksize, self.sm_count
                )
            return (
                pool_seqlens,
                pool_context_lens,
                pool_block_tables,
                pool_schedule_metadata,
            )

        pool_seqlens = attn_metadata.pooled_cache_seqlens_int32
        pool_block_tables = attn_metadata.pooled_real_page_table
        pool_schedule_metadata = attn_metadata.pooled_paged_mqa_schedule_metadata

        if (
            pool_seqlens is None
            or pool_block_tables is None
            or attn_metadata.pooled_index_kpool != self.index_kpool
        ):
            pool_seqlens = torch.div(
                seqlens_32, self.index_kpool, rounding_mode="floor"
            ).to(torch.int32)
            pool_block_tables = build_pooled_page_table_64(
                block_tables, self.index_kpool
            ).contiguous()
            pool_schedule_metadata = None
        else:
            pool_seqlens = pool_seqlens[: seqlens_32.shape[0]]
            pool_block_tables = pool_block_tables[
                : block_tables.shape[0],
                : (block_tables.shape[1] + self.index_kpool - 1) // self.index_kpool,
            ]

        pool_context_lens = pool_seqlens.contiguous().view(-1, 1)
        if pool_schedule_metadata is None and build_schedule_metadata:
            pool_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                pool_context_lens.clamp(min=1), blocksize, self.sm_count
            )

        return (
            pool_seqlens,
            pool_context_lens,
            pool_block_tables,
            pool_schedule_metadata,
        )

    @staticmethod
    def _kpool_fused_topk_mapping(
        metadata: BaseIndexerMetadata,
        paged_page_table: Optional[torch.Tensor] = None,
        paged_page_table_row_index: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not envs.SGLANG_DSA_FUSE_TOPK.get():
            return None, None, None

        topk_method = metadata.topk_transform_method
        attn_metadata = metadata.attn_metadata
        if topk_method == TopkTransformMethod.PAGED:
            page_table_1 = (
                paged_page_table
                if paged_page_table is not None
                else attn_metadata.page_table_1
            )
            assert page_table_1 is not None
            row_index = paged_page_table_row_index
            return page_table_1, None, row_index
        if topk_method == TopkTransformMethod.RAGGED:
            return None, attn_metadata.topk_indices_offset, None
        return None, None, None

    @staticmethod
    def _should_use_tilelang_paged_mqa_logits(q_fp8: torch.Tensor) -> bool:
        if not is_cuda():
            return False
        arch_major, _ = torch.cuda.get_device_capability(q_fp8.device)
        num_heads = q_fp8.shape[1]
        return arch_major == 9 and num_heads not in (32, 64)

    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        pool = get_token_to_kv_pool()
        page_size = pool.page_size
        # DeepGEMM paged-MQA requires 64-token pages.
        assert page_size == 64, "only support page size 64"

        block_tables = metadata.get_page_table_64()

        kv_cache_fp8 = self._get_index_k_read_buffer(pool, layer_id)

        blocksize = page_size
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()
        assert len(q_fp8.shape) == 3
        num_q_padded = q_fp8.shape[0]
        n_real = seqlens_32.shape[0]
        if n_real < num_q_padded:
            q_fp8 = q_fp8[:n_real]
            weights = weights[:n_real]
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        use_aiter_paged_mqa = is_hip()
        use_tilelang_paged_mqa = (
            not use_aiter_paged_mqa
            and self._should_use_tilelang_paged_mqa_logits(q_fp8)
        )

        pool_seqlens, pool_context_lens, pool_block_tables, pool_schedule_metadata = (
            self._get_kpool_decode_metadata(
                metadata,
                block_tables,
                seqlens_32,
                blocksize,
                build_schedule_metadata=not (
                    use_aiter_paged_mqa or use_tilelang_paged_mqa
                ),
            )
        )
        pool_max_seq_len = pool_block_tables.shape[1] * blocksize
        if use_aiter_paged_mqa:
            if not aiter_can_use_preshuffle_paged_mqa():
                raise RuntimeError(
                    "ROCm kpool indexer requires the AITER preshuffle paged-MQA kernel"
                )
            from sglang.kernels.ops.attention.dsa import aiter_paged_mqa_logits

            logits = aiter_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                pool_seqlens,
                pool_block_tables,
                pool_max_seq_len,
                preshuffle=True,
                kv_block_size=block_kv,
            )
        elif use_tilelang_paged_mqa:
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits,
            )

            logits = tilelang_fp8_paged_mqa_logits(
                q_fp8.unsqueeze(1),
                kv_cache_fp8,
                weights,
                pool_seqlens,
                pool_block_tables,
                pool_schedule_metadata,
                pool_max_seq_len,
                clean_logits=False,
            )
        else:
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8.unsqueeze(1),
                kv_cache_fp8,
                weights,
                pool_context_lens,
                pool_block_tables,
                pool_schedule_metadata,
                pool_max_seq_len,
                clean_logits=False,
            )

        page_table_1, topk_offsets, _ = self._kpool_fused_topk_mapping(metadata)
        topk_result = self._topk_from_kpool_logits(
            logits,
            pool_seqlens,
            seq_lens=seqlens_32,
            page_table=page_table_1,
            topk_offsets=topk_offsets,
            out_rows=num_q_padded if num_q_padded != n_real else None,
        )
        return topk_result

    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device: torch.device
    ) -> Tuple[bool, int]:
        if num_q * num_k < 8_000_000:
            return False, 0

        free_mem, total_mem = torch.cuda.mem_get_info(device)
        bytes_per_elem = 4
        logits_bytes = num_q * num_k * bytes_per_elem

        need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
        return need_chunk, free_mem

    def _get_topk_ragged_kpool_plan(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            gather_index_k_scale_prefix_into,
        )

        plan = metadata.attn_metadata.kpool_extend_plan
        assert plan is not None, "kpool extend plan is required"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        device = q_fp8.device
        total_q = q_fp8.shape[0]
        seq_lens_expanded = plan.seq_lens_expanded
        pool_lens = plan.pooled_seq_lens_expanded
        ks_per_q = plan.ragged_q_ks
        ke_per_q = plan.ragged_q_ke
        total_k_rows = plan.ragged_total_k_rows

        n_real = seq_lens_expanded.shape[0]
        row_select = None
        if n_real > total_q:
            row_select = cp_zigzag_full_plan_rows(forward_batch, device)
            if row_select is not None:
                seq_lens_expanded = seq_lens_expanded.index_select(0, row_select)
                pool_lens = pool_lens.index_select(0, row_select)
                ks_per_q = ks_per_q.index_select(0, row_select)
                ke_per_q = ke_per_q.index_select(0, row_select)
                n_real = seq_lens_expanded.shape[0]
        assert (
            n_real <= total_q
        ), f"plan has more real rows ({n_real}) than q_fp8 ({total_q})"

        if total_k_rows > 0:
            k_u8 = plan.ragged_k_u8
            k_scale = plan.ragged_k_scale
            assert k_u8 is not None and k_scale is not None
            pool = get_token_to_kv_pool()
            gather_index_k_scale_prefix_into(
                pool=pool,
                buf=self._get_index_k_read_buffer(pool, layer_id),
                page_indices=plan.ragged_concat_page_table,
                seq_len=total_k_rows,
                k_out=k_u8,
                scale_out=k_scale,
            )
            k_fp8 = k_u8.view(torch.float8_e4m3fn)
            logits = self._fp8_mqa_logits(
                q_fp8[:n_real].contiguous(),
                k_fp8.contiguous(),
                k_scale.contiguous(),
                weights[:n_real].contiguous(),
                ks_per_q,
                ke_per_q,
                clean_logits=True,
            )
        else:
            logits = torch.empty((n_real, 0), dtype=torch.float32, device=device)

        topk_method = metadata.topk_transform_method
        attn_metadata = metadata.attn_metadata
        page_table_all = None
        page_table_row_index_all = None
        topk_offsets_all = None
        if envs.SGLANG_DSA_FUSE_TOPK.get():
            if topk_method == TopkTransformMethod.PAGED:
                page_table_all = plan.ragged_paged_page_table
                page_table_row_index_all = plan.ragged_paged_page_table_row_index
                if page_table_row_index_all is not None and row_select is not None:
                    page_table_row_index_all = page_table_row_index_all.index_select(
                        0, row_select
                    )
                elif page_table_all is not None and row_select is not None:
                    page_table_all = page_table_all.index_select(0, row_select)
            elif topk_method == TopkTransformMethod.RAGGED:
                topk_offsets_all = attn_metadata.topk_indices_offset
                if topk_offsets_all is not None and row_select is not None:
                    topk_offsets_all = topk_offsets_all.index_select(0, row_select)

        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=seq_lens_expanded,
            page_table=page_table_all,
            topk_offsets=topk_offsets_all,
            row_starts=ks_per_q,
            out_rows=total_q,
            page_table_row_index=page_table_row_index_all,
        )

    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: Optional[List[Tuple[int, int, int]]] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            build_pooled_page_table_64,
            gather_index_k_scale_prefix_into,
        )

        assert cp_index is None, "DSA kpool CP topk currently supports batch size 1"
        assert forward_batch.batch_size == 1
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        pool = get_token_to_kv_pool()
        pool_size = self.index_kpool
        slots_per_page = pool.slots_per_page
        device = q_fp8.device
        out_rows = q_fp8.shape[0]
        actual_seq_q = int(actual_seq_q)

        assert forward_batch.seq_lens_cpu is not None
        assert forward_batch.extend_seq_lens_cpu is not None
        kv_len_token = int(
            forward_batch.seq_lens_cpu[0].item()
            - forward_batch.extend_seq_lens_cpu[0]
            + kv_len
        )
        pool_kv_len = kv_len_token // pool_size

        q_work = q_fp8[:actual_seq_q].contiguous()
        weights_work = weights[:actual_seq_q].contiguous()
        tail_tokens = torch.arange(
            kv_len_token - actual_seq_q + 1,
            kv_len_token + 1,
            dtype=torch.int32,
            device=device,
        )

        if pool_kv_len > 0 and actual_seq_q > 0:
            block_tables = metadata.get_page_table_64()
            bt_row = block_tables[0]
            n_pages = (pool_kv_len + slots_per_page - 1) // slots_per_page
            packed_page_indices = (
                build_pooled_page_table_64(bt_row, pool_size)[:n_pages]
                .to(torch.int32)
                .contiguous()
            )
            k_u8 = torch.empty(
                (pool_kv_len, self.head_dim), dtype=torch.uint8, device=device
            )
            k_scale = torch.empty((pool_kv_len,), dtype=torch.float32, device=device)
            gather_index_k_scale_prefix_into(
                pool=pool,
                buf=self._get_index_k_read_buffer(pool, layer_id),
                page_indices=packed_page_indices,
                seq_len=pool_kv_len,
                k_out=k_u8,
                scale_out=k_scale,
            )
            k_fp8 = k_u8.view(torch.float8_e4m3fn)
            ks = torch.zeros((actual_seq_q,), dtype=torch.int32, device=device)
            ke = torch.div(tail_tokens, pool_size, rounding_mode="floor").to(
                torch.int32
            )
            logits = self._fp8_mqa_logits(
                q_work,
                k_fp8.contiguous(),
                k_scale.contiguous(),
                weights_work,
                ks,
                ke,
                clean_logits=True,
            )
            pool_lens = ke
        else:
            logits = torch.empty((actual_seq_q, 0), dtype=torch.float32, device=device)
            pool_lens = torch.zeros((actual_seq_q,), dtype=torch.int32, device=device)
            ks = torch.zeros((actual_seq_q,), dtype=torch.int32, device=device)

        page_table_local = None
        topk_method = metadata.topk_transform_method
        if envs.SGLANG_DSA_FUSE_TOPK.get() and topk_method == TopkTransformMethod.PAGED:
            req_pool_idx = int(forward_batch.req_pool_indices[0].item())
            page_table_local = (
                get_req_to_token_pool()
                .req_to_token[req_pool_idx, :kv_len_token]
                .to(torch.int32)
                .unsqueeze(0)
                .expand(actual_seq_q, -1)
            )

        return self._topk_from_kpool_logits(
            logits,
            pool_lens,
            seq_lens=tail_tokens,
            page_table=page_table_local,
            topk_offsets=None,
            row_starts=ks,
            out_rows=out_rows,
        )

    def _get_topk_ragged_kpool(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        extend_pooled_cache: Optional[
            List[Optional[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]]
        ] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            build_pooled_page_table_64,
            gather_index_k_scale_prefix_into,
        )

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        pool_size = self.index_kpool
        page_size = get_token_to_kv_pool().page_size
        token_nums = q_fp8.shape[0]
        tail_pool = pool_size - 1
        topk_result = torch.empty(
            (token_nums, self.index_topk + tail_pool),
            device=q_fp8.device,
            dtype=torch.int32,
        )
        block_tables = metadata.get_page_table_64()
        seq_lens_expanded = metadata.get_seqlens_expanded()
        topk_method = metadata.topk_transform_method
        attn_metadata = metadata.attn_metadata
        topk_offsets = attn_metadata.topk_indices_offset
        pooled_page_tables = getattr(
            attn_metadata, "kpool_extend_pooled_page_tables", None
        )
        zero_starts_by_batch = getattr(attn_metadata, "kpool_extend_zero_starts", None)
        pooled_seq_lens_expanded = getattr(
            attn_metadata, "kpool_seqlens_expanded", None
        )

        q_offset = 0
        cache_write_stream = None
        pending_cache_req_pool_indices = set()
        pending_cache_tensors = []
        for i in range(forward_batch.batch_size):
            q_len = int(forward_batch.extend_seq_lens_cpu[i])
            if q_len == 0:
                continue

            seq_len = int(forward_batch.seq_lens_cpu[i].item())
            pool_seq_len = seq_len // pool_size
            req_pool_idx = int(forward_batch.req_pool_indices[i].item())
            if (
                cache_write_stream is not None
                and req_pool_idx in pending_cache_req_pool_indices
            ):
                torch.cuda.current_stream().wait_stream(cache_write_stream)
                cache_write_stream = None
                pending_cache_req_pool_indices.clear()
                pending_cache_tensors.clear()

            q_slice = slice(q_offset, q_offset + q_len)
            local_seqlens = seq_lens_expanded[q_slice]
            if pooled_seq_lens_expanded is None:
                local_pool_lens = torch.div(
                    local_seqlens, pool_size, rounding_mode="floor"
                ).to(torch.int32)
            else:
                local_pool_lens = pooled_seq_lens_expanded[q_slice]

            if pool_seq_len > 0:
                cached_current = (
                    extend_pooled_cache[i] if extend_pooled_cache is not None else None
                )
                deferred_cache_write = None
                if cached_current is not None:
                    curr_pool_start, curr_k_fp8, curr_k_scale = cached_current[:3]
                    curr_pool_len = curr_k_fp8.shape[0]
                    if len(cached_current) == 4:
                        deferred_cache_write = (
                            cached_current[3],
                            curr_k_fp8,
                            curr_k_scale,
                        )
                else:
                    curr_pool_start, curr_pool_len = pool_seq_len, 0
                if (
                    pool_seq_len > 0
                    and deferred_cache_write is not None
                    and self.alt_stream is not None
                    and not get_is_capture_mode()
                ):
                    write_locs, write_k_fp8, write_k_scale = deferred_cache_write
                    current_stream = torch.cuda.current_stream()
                    self.alt_stream.wait_stream(current_stream)
                    pending_cache_tensors.append(
                        (write_locs, write_k_fp8, write_k_scale)
                    )
                    with torch.cuda.stream(self.alt_stream):
                        self._write_returned_compressed_pooled_index_cache(
                            forward_batch,
                            layer_id,
                            write_locs,
                            write_k_fp8,
                            write_k_scale,
                        )
                    cache_write_stream = self.alt_stream
                    pending_cache_req_pool_indices.add(req_pool_idx)
                    deferred_cache_write = None

                if (
                    cached_current is not None
                    and curr_pool_start == 0
                    and curr_pool_len == pool_seq_len
                ):
                    k_fp8 = curr_k_fp8
                    k_scale = curr_k_scale
                elif (
                    cached_current is not None
                    and curr_pool_start >= 0
                    and curr_pool_start + curr_pool_len == pool_seq_len
                ):
                    if curr_pool_start > 0:
                        pooled_page_table = (
                            pooled_page_tables[i]
                            if pooled_page_tables is not None
                            else None
                        )
                        if pooled_page_table is None:
                            num_token_pages = (seq_len + page_size - 1) // page_size
                            token_page_table = block_tables[
                                i, :num_token_pages
                            ].contiguous()
                            pool_pages = (curr_pool_start + page_size - 1) // page_size
                            pooled_page_table = build_pooled_page_table_64(
                                token_page_table, pool_size
                            )[:pool_pages].contiguous()
                        k_u8 = torch.empty(
                            (pool_seq_len, self.head_dim),
                            dtype=torch.uint8,
                            device=q_fp8.device,
                        )
                        k_scale = torch.empty(
                            (pool_seq_len,), dtype=torch.float32, device=q_fp8.device
                        )
                        pool = get_token_to_kv_pool()
                        gather_index_k_scale_prefix_into(
                            pool=pool,
                            buf=self._get_index_k_read_buffer(pool, layer_id),
                            page_indices=pooled_page_table,
                            seq_len=curr_pool_start,
                            k_out=k_u8,
                            scale_out=k_scale,
                        )
                        k_u8[curr_pool_start:pool_seq_len].copy_(
                            curr_k_fp8.view(torch.uint8)
                        )
                        k_scale[curr_pool_start:pool_seq_len].copy_(curr_k_scale)
                        k_fp8 = k_u8.view(torch.float8_e4m3fn)
                    else:
                        k_fp8 = curr_k_fp8
                        k_scale = curr_k_scale
                else:
                    pooled_page_table = (
                        pooled_page_tables[i]
                        if pooled_page_tables is not None
                        else None
                    )
                    if pooled_page_table is None:
                        num_token_pages = (seq_len + page_size - 1) // page_size
                        token_page_table = block_tables[
                            i, :num_token_pages
                        ].contiguous()
                        pool_pages = (pool_seq_len + page_size - 1) // page_size
                        pooled_page_table = build_pooled_page_table_64(
                            token_page_table, pool_size
                        )[:pool_pages].contiguous()
                    seq_len_t = torch.tensor(
                        [pool_seq_len], dtype=torch.int32, device=q_fp8.device
                    )
                    k_fp8, k_scale = get_token_to_kv_pool().get_index_k_scale_buffer(
                        layer_id,
                        seq_len_t,
                        pooled_page_table.unsqueeze(0),
                        pool_seq_len,
                        pool_seq_len,
                    )
                    k_fp8 = k_fp8.view(torch.float8_e4m3fn)
                    k_scale = k_scale.view(torch.float32).squeeze(-1)
                row_starts = (
                    zero_starts_by_batch[i]
                    if zero_starts_by_batch is not None
                    and zero_starts_by_batch[i] is not None
                    else torch.zeros((q_len,), dtype=torch.int32, device=q_fp8.device)
                )
                local_logits = self._fp8_mqa_logits(
                    q_fp8[q_slice].contiguous(),
                    k_fp8.contiguous(),
                    k_scale.contiguous(),
                    weights[q_slice].contiguous(),
                    row_starts,
                    local_pool_lens,
                    clean_logits=True,
                )
            else:
                local_logits = torch.empty(
                    (q_len, 0), dtype=torch.float32, device=q_fp8.device
                )

            page_table_local = None
            topk_offsets_local = None
            if (
                envs.SGLANG_DSA_FUSE_TOPK.get()
                and topk_method == TopkTransformMethod.PAGED
            ):
                page_table_local = (
                    get_req_to_token_pool()
                    .req_to_token[req_pool_idx, :seq_len]
                    .to(torch.int32)
                )
                page_table_local = page_table_local.unsqueeze(0).expand(q_len, -1)
            elif (
                envs.SGLANG_DSA_FUSE_TOPK.get()
                and topk_method == TopkTransformMethod.RAGGED
                and topk_offsets is not None
            ):
                topk_offsets_local = topk_offsets[q_slice]

            local_topk = self._topk_from_kpool_logits(
                local_logits,
                local_pool_lens,
                seq_lens=local_seqlens,
                page_table=page_table_local,
                topk_offsets=topk_offsets_local,
            )

            topk_result[q_slice] = local_topk
            if pool_seq_len > 0 and deferred_cache_write is not None:
                write_locs, curr_k_fp8, curr_k_scale = deferred_cache_write
                self._write_returned_compressed_pooled_index_cache(
                    forward_batch,
                    layer_id,
                    write_locs,
                    curr_k_fp8,
                    curr_k_scale,
                )
            q_offset += q_len

        if cache_write_stream is not None:
            torch.cuda.current_stream().wait_stream(cache_write_stream)
            pending_cache_tensors.clear()

        return topk_result

    def _get_topk_ragged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kpool_extend_cache: Optional[
            List[Optional[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]]
        ] = None,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        assert forward_batch.forward_mode.is_extend_without_speculative()

        page_size = get_token_to_kv_pool().page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        if metadata.attn_metadata.kpool_extend_plan is not None:
            return self._get_topk_ragged_kpool_plan(
                forward_batch,
                layer_id,
                q_fp8,
                weights,
                metadata,
            )
        return self._get_topk_ragged_kpool(
            forward_batch,
            layer_id,
            q_fp8,
            weights,
            metadata,
            extend_pooled_cache=kpool_extend_cache,
        )

    def _forward_cuda_skip_logits(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert forward_batch.forward_mode.is_extend_without_speculative()

        key = self._get_k_bf16(x, positions)
        self._compress_write(
            x=x,
            key=key,
            positions=positions,
            forward_batch=forward_batch,
            layer_id=layer_id,
            metadata=metadata,
        )

        if not return_indices:
            return None

        return self._full_topk_for_short_sequence(metadata, x.device)

    def _forward_cuda_target_verify(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        metadata: BaseIndexerMetadata,
        enable_dual_stream: bool,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert is_cuda(), "DSA kpool target_verify is CUDA-only"
        plan = metadata.attn_metadata.kpool_write_plan
        assert plan is not None, "DSA kpool target_verify requires kpool_write_plan"
        num_draft_tokens = plan.num_draft_tokens

        query, key, gate_score_maybe = self._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream=enable_dual_stream,
            forward_batch=forward_batch,
            precompute_compress_gate=(
                enable_dual_stream and self.compress_gate_stream is not None
            ),
        )

        pool = get_token_to_kv_pool()
        tail_k_buf, tail_score_buf = pool.get_compress_tail_buffers(layer_id)

        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            kpool_write_tail_and_maybe_compress,
        )

        buf = pool.get_index_k_with_scale_buffer(layer_id=layer_id)

        def _compress_write() -> None:
            kpool_write_tail_and_maybe_compress(
                pool=pool,
                buf=buf,
                key=key,
                score=self._compute_gate_score_if_missing(x, gate_score_maybe),
                tail_k=tail_k_buf,
                tail_score=tail_score_buf,
                ape=self.index_kpool_compress_ape,
                req_pool_indices=plan.req,
                write_start=plan.write_start,
                tail_logical_start=plan.tail_logical_start,
                write_loc=plan.write_loc,
                out_cache_loc=forward_batch.out_cache_loc,
                num_draft_tokens=num_draft_tokens,
                round_scale=self.scale_fmt is not None,
                effective_n_per_batch=plan.effective_n_per_batch,
            )

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if gate_score_maybe is not None:
                assert self.compress_gate_stream is not None
                self.alt_stream.wait_stream(self.compress_gate_stream)
            if return_indices:
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                weights = self._get_logits_head_gate(x, q_scale)
            with torch.cuda.stream(self.alt_stream):
                _compress_write()
            current_stream.wait_stream(self.alt_stream)
        else:
            _compress_write()
            if return_indices:
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                weights = self._get_logits_head_gate(x, q_scale)

        if not return_indices:
            return None
        return self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if is_hip():
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant
        elif not is_npu():
            from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # Skip DSA if the attention backend chooses to skip this batch.
        if metadata is None:
            return None

        assert forward_batch.seq_lens_cpu is not None
        mode = forward_batch.forward_mode
        if mode.is_idle() or len(forward_batch.seq_lens_cpu) == 0:
            return torch.full(
                (x.shape[0], self.index_topk + self.index_kpool - 1),
                -1,
                dtype=torch.int,
                device=x.device,
            )

        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return self._forward_cuda_target_verify(
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                act_quant=act_quant,
                metadata=metadata,
                enable_dual_stream=enable_dual_stream,
                return_indices=return_indices,
            )

        skip_logits_computation = False
        if forward_batch.forward_mode.is_extend_without_speculative():
            if forward_batch.seq_lens_cpu is not None:
                max_kv_len = forward_batch.seq_lens_cpu.max().item()
                skip_logits_computation = max_kv_len <= self.index_topk

        if skip_logits_computation and (not self.dsa_enable_prefill_cp):
            return self._forward_cuda_skip_logits(
                x,
                positions,
                forward_batch,
                layer_id,
                act_quant,
                metadata,
                return_indices,
            )

        precompute_compress_gate = (
            self.index_kpool > 1
            and self.index_kpool_compress
            and enable_dual_stream
            and forward_batch.forward_mode.is_decode_or_idle()
            and self.compress_gate_stream is not None
        )
        query, key, gate_score = self._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream,
            forward_batch=forward_batch,
            precompute_compress_gate=precompute_compress_gate,
        )
        use_cp = dsa_use_prefill_cp(forward_batch, self.dsa_enable_prefill_cp)
        if use_cp:
            if gate_score is None:
                gate_score = self._compute_gate_score_if_missing(x, gate_score)
            key, gate_score = self._cp_gather_concat(
                [key, gate_score], self.cp_size, forward_batch
            )

        weights = None
        kpool_extend_cache = None
        if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            if gate_score is not None:
                self.alt_stream.wait_stream(self.compress_gate_stream)
            with torch.cuda.stream(self.alt_stream):
                self._compress_write(
                    x=x,
                    key=key,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    metadata=metadata,
                    gate_score=gate_score,
                )
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            weights = self._get_logits_head_gate(x, q_scale)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            has_kpool_extend_plan = metadata.attn_metadata.kpool_extend_plan is not None
            defer_kpool_cache_write = (
                forward_batch.forward_mode.is_extend_without_speculative()
                and return_indices
                and not has_kpool_extend_plan
            )
            kpool_extend_cache = self._compress_write(
                x=x,
                key=key,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=layer_id,
                metadata=metadata,
                gate_score=gate_score,
                return_compressed=(
                    forward_batch.forward_mode.is_extend_without_speculative()
                    and return_indices
                ),
                write_cache=not defer_kpool_cache_write,
            )
            if (
                forward_batch.forward_mode.is_extend_without_speculative()
                and not return_indices
            ):
                return None

        if weights is None:
            weights = self._get_logits_head_gate(x, q_scale)

        if is_cuda() or is_hip():
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend_v2()
            ):
                topk_result = self._get_topk_paged(
                    forward_batch, layer_id, q_fp8, weights, metadata
                )
            else:
                if (
                    forward_batch.attn_cp_metadata is not None
                    and self.dsa_enable_prefill_cp
                    and is_dsa_prefill_cp_in_seq_split()
                ):
                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev_list[0]
                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next_list[0]
                    actual_seq_q_prev = (
                        forward_batch.attn_cp_metadata.actual_seq_q_prev_list[0]
                    )
                    actual_seq_q_next = (
                        forward_batch.attn_cp_metadata.actual_seq_q_next_list[0]
                    )
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )
                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    topk_result = torch.cat([topk_result_prev, topk_result_next], dim=0)
                elif has_kpool_extend_plan:
                    topk_result = self._get_topk_ragged_kpool_plan(
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                    )
                else:
                    topk_result = self._get_topk_ragged(
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                        kpool_extend_cache=kpool_extend_cache,
                    )
        else:
            raise NotImplementedError("kpool indexer is only supported on CUDA")
        return topk_result
